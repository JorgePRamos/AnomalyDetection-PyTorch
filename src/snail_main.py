import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
from models.pixelSnail import PixelSNAIL
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from optim import Adam


# Parameters
start_epoch = 0
n_epochs = 5
batch_size = 16
step = 0
output_dir = Path("E:/")
n_samples = 1 #8

device = "cuda"
image_dims = [50,16,16]
n_channels = 50

#default training values
n_res_layers = 5
attn_n_layers = 12
attn_nh = 1
attn_dq = 16
attn_dv = 128
attn_drop_rate = 0
n_logistic_mix = 10
n_cond_classes = 0

polyak = 0.9995
lr = 5e-4
lr_decay = 0.999995

n_bits = 4
log_interval = 1
eval_interval = 1 #10





# Define your dataset class
class EncodingsDataset(Dataset):
    def __init__(self, rootDir, train = True):
        self.rootDir = rootDir
        if train:
            encDir = rootDir / r'train/'
        self.encDir = encDir
        self.encList = sorted(glob.glob(os.path.join(encDir, '**/*.npy')))

    def __len__(self):
        return len(self.encList)

    def __getitem__(self, idx):
        enc = np.load(self.encList[idx])

        return torch.from_numpy(enc),self.encList[idx]

def discretized_mix_logistic_loss(l, x, n_bits):
    """ log likelihood for mixture of discretized logistics
    Args
        l -- model output tensor of shape (B, 10*n_mix, H, W), where for each n_mix there are
                3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        x -- data tensor of shape (B, C, H, W) with values in model space [-1, 1]
    """
    # shapes
    B, C, H, W = x.shape
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]                         # (B, n_mix, H, W)
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)        # (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
    x  = x.unsqueeze(1).expand_as(means)
    if C!=1:
        m1 = means[:, :, 0, :, :]
        m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * x[:, :, 0, :, :]
        m3 = means[:, :, 2, :, :] + coeffs[:, :, 1, :, :] * x[:, :, 0, :, :] + coeffs[:, :, 2, :, :] * x[:, :, 1, :, :]
        means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

    # log prob components
    scales = torch.exp(-logscales)
    plus = scales * (x - means + 1/(2**n_bits-1))
    minus = scales * (x - means - 1/(2**n_bits-1))

    # partition the logistic pdf and cdf for x in [<-0.999, mid, >0.999]
    # 1. x<-0.999 ie edge case of 0 before scaling
    cdf_minus = torch.sigmoid(minus)
    log_one_minus_cdf_minus = - F.softplus(minus)
    # 2. x>0.999 ie edge case of 255 before scaling
    cdf_plus = torch.sigmoid(plus)
    log_cdf_plus = plus - F.softplus(plus)
    # 3. x in [-.999, .999] is log(cdf_plus - cdf_minus)

    # compute log probs:
    # 1. for x < -0.999, return log_cdf_plus
    # 2. for x > 0.999,  return log_one_minus_cdf_minus
    # 3. x otherwise,    return cdf_plus - cdf_minus
    log_probs = torch.where(x < -0.999, log_cdf_plus,
                            torch.where(x > 0.999, log_one_minus_cdf_minus,
                                        torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))))
    log_probs = log_probs.sum(2) + F.log_softmax(logits, 1) # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

    # marginalize over n_mix components and return negative log likelihood per data point
    return - log_probs.logsumexp(1).sum([1,2])  # out (B,)




def sample_from_discretized_mix_logistic(l, image_dims):
    # shapes
    B, _, H, W = l.shape
    C = image_dims[0]#3
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)  # each out (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # sample mixture indicator
    argmax = torch.argmax(logits - torch.log(-torch.log(torch.rand_like(logits).uniform_(1e-5, 1 - 1e-5))), dim=1)
    sel = torch.eye(n_mix, device=logits.device)[argmax]
    sel = sel.permute(0,3,1,2).unsqueeze(2)  # (B, n_mix, 1, H, W)

    # select mixture components
    means = means.mul(sel).sum(1)
    logscales = logscales.mul(sel).sum(1)
    coeffs = coeffs.mul(sel).sum(1)

    # sample from logistic using inverse transform sampling
    u = torch.rand_like(means).uniform_(1e-5, 1 - 1e-5)
    x = means + logscales.exp() * (torch.log(u) - torch.log1p(-u))  # logits = inverse logistic

    if C==1:
        return x.clamp(-1,1)
    else:
        x0 = torch.clamp(x[:,0,:,:], -1, 1)
        x1 = torch.clamp(x[:,1,:,:] + coeffs[:,0,:,:] * x0, -1, 1)
        x2 = torch.clamp(x[:,2,:,:] + coeffs[:,1,:,:] * x0 + coeffs[:,2,:,:] * x1, -1, 1)
        return torch.stack([x0, x1, x2], 1)  # out (B, C, H, W)
    
def generate_fn(model, n_samples, image_dims, device, h=None):
    out = torch.zeros(n_samples, *image_dims, device=device)
    with tqdm(total=(image_dims[1]*image_dims[2]), desc='Generating {} images'.format(n_samples)) as pbar:
        for yi in range(image_dims[1]):
            for xi in range(image_dims[2]):
                l = model(out, h)
                out[:,:,yi,xi] = sample_from_discretized_mix_logistic(l, image_dims)[:,:,yi,xi]
                pbar.update()
    return out


def fetch_dataloaders():
    # Load your dataset
    rootDir = Path("E:/mvtec_encodings/bottle/")
    dataset = EncodingsDataset(rootDir)

    train_dataloader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=(device.type=='cuda'), num_workers=4)
    #valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, pin_memory=(device.type=='cuda'), num_workers=4)
    return train_dataloader

# Training loop
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, start_epoch + n_epochs)) as pbar:
        for x,y in dataloader:
            step += 1

            x = x.to(device)
            logits = model(x, y.to(device) if n_cond_classes else None)
            loss = loss_fn(logits, x, n_bits).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            pbar.set_postfix(bits_per_dim='{:.4f}'.format(loss.item() / (np.log(2) * np.prod(image_dims))))
            pbar.update()

            # record
            """if step % log_interval == 0:
                writer.add_scalar('train_bits_per_dim', loss.item() / (np.log(2) * np.prod(image_dims)), step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)"""  

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses = 0
    for x,y in tqdm(dataloader, desc='Evaluate'):
        x = x.to(device)
        logits = model(x, y.to(device) if n_cond_classes else None)
        losses += loss_fn(logits, x, n_bits).mean(0).item()
    return losses / len(dataloader)

@torch.no_grad()
def generate(model, generate_fn, args):
    model.eval()
    if n_cond_classes:
        samples = []
        for h in range(n_cond_classes):
            h = torch.eye(n_cond_classes)[h,None].to(device)
            samples += [generate_fn(model, n_samples, image_dims, device, h=h)]
        samples = torch.cat(samples)
    else:
        samples = generate_fn(model, n_samples, image_dims, device)
    return make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=n_samples)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, args):
    for epoch in range(start_epoch, start_epoch + n_epochs):
        # train
        train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, args)

        if (epoch+1) % eval_interval == 0:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': step,
                        'state_dict': model.state_dict()},
                        os.path.join(output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(output_dir, 'sched_checkpoint.pt'))

            # swap params to ema values
            optimizer.swap_ema()

            # evaluate
            eval_loss = evaluate(model, test_dataloader, loss_fn, args)
            print('Evaluate bits per dim: {:.3f}'.format(eval_loss.item() / (np.log(2) * np.prod(image_dims))))
            #writer.add_scalar('eval_bits_per_dim', eval_loss.item() / (np.log(2) * np.prod(image_dims)), step)

            # generate
            samples = generate(model, generate_fn, args)
            #writer.add_image('samples', samples, step)
            save_image(samples, os.path.join(output_dir, 'generation_sample_step_{}.png'.format(step)))

            # restore params to gradient optimized
            optimizer.swap_ema()


if __name__ == '__main__':

   

    model = PixelSNAIL(image_dims, n_channels, n_res_layers, attn_n_layers, attn_nh, attn_dq, attn_dv, attn_drop_rate, n_logistic_mix, n_cond_classes).to(device)
    loss_fn = discretized_mix_logistic_loss
    loss_fn = loss_fn
    generate_fn = generate_fn
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.95, 0.9995), polyak=polyak, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    # Model Summary
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))



    train_dataloader, test_dataloader = fetch_dataloaders()


    


    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn)
