# %% [markdown]
# # Visualizing latent space with t-SNE

# %% [markdown]
# ## Initialization

# %%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("/media/hdd/viscent/SR-UNet")
from utils.inference_utils import *

# %%
# %%capture
num_samples = 50
dhcp_train_loader, dhcp_test_loader, dhcp_val_loader = get_dataloader('dhcp', num_samples=num_samples,modality='t2')
hcp_train_loader, hcp_test_loader, hcp_val_loader = get_dataloader('hcp', num_samples=num_samples,modality='t2')
brats_train_loader, brats_test_loader, brats_val_loader = get_dataloader('brats', num_samples=num_samples,modality='t2')

# %%
# %%capture 
model_dhcp = load_model('/media/hdd/viscent/SR-UNet/pretrained_models/dHCP/best.pth')
model_hcp = load_model('/media/hdd/viscent/SR-UNet/pretrained_models/HCP/best.pth')
model_brats = load_model('/media/hdd/viscent/SR-UNet/pretrained_models/BraTS/best.pth')
model_dhcp = model_dhcp.cuda()
model_hcp = model_hcp.cuda()
model_brats = model_brats.cuda()
model_dhcp.eval()
model_hcp.eval()
model_brats.eval()
model = model_dhcp

# %%
# dHCP Model
latent_dhcp_dhcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_dhcp_train.pt')
latent_hcp_dhcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_dhcp_train.pt')
latent_brats_dhcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_dhcp_train.pt')
latent_dhcp_dhcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_dhcp_val.pt')
latent_hcp_dhcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_dhcp_val.pt')
latent_brats_dhcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_dhcp_val.pt')
latent_brown_dhcp = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brown_dhcp.pt')
# HCP Model
latent_dhcp_hcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_hcp_train.pt')
latent_hcp_hcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_hcp_train.pt')
latent_brats_hcp_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_hcp_train.pt')
latent_dhcp_hcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_hcp_val.pt')
latent_hcp_hcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_hcp_val.pt')
latent_brats_hcp_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_hcp_val.pt')
latent_brown_hcp = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brown_hcp.pt')
# BraTS Model
latent_dhcp_brats_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_brats_train.pt')
latent_hcp_brats_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_brats_train.pt')
latent_brats_brats_train = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_brats_train.pt')
latent_dhcp_brats_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp_brats_val.pt')
latent_hcp_brats_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp_brats_val.pt')
latent_brats_brats_val = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brats_brats_val.pt')
latent_brown_brats = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brown_brats.pt')

# %%
divergences_innate = np.zeros(3)
divergences_innate[0] = SamplesLoss("sinkhorn")(latent_brats_brats_val.flatten(start_dim=1), latent_brats_brats_train.flatten(start_dim=1))
divergences_innate[1] = SamplesLoss("sinkhorn")(latent_hcp_hcp_val.flatten(start_dim=1), latent_hcp_hcp_train.flatten(start_dim=1))
divergences_innate[2] = SamplesLoss("sinkhorn")(latent_dhcp_dhcp_val.flatten(start_dim=1), latent_dhcp_dhcp_train.flatten(start_dim=1))
latents = [latent_brats_brats_train,latent_hcp_hcp_train,latent_dhcp_dhcp_train]

# %% [markdown]
# # t-SNE

# %% [markdown]
# ## dHCP Model

# %%
cursors = [0]
cursors.append(cursors[-1] + latent_dhcp_dhcp_train.shape[0])
cursors.append(cursors[-1] + latent_dhcp_dhcp_val.shape[0])
cursors.append(cursors[-1] + latent_hcp_dhcp_train.shape[0])
cursors.append(cursors[-1] + latent_hcp_dhcp_val.shape[0])
cursors.append(cursors[-1] + latent_brats_dhcp_train.shape[0])
cursors.append(cursors[-1] + latent_brats_dhcp_val.shape[0])
cursors.append(cursors[-1] + latent_brown_dhcp.shape[0])
labels = torch.zeros(cursors[-1])
labels[cursors[0]:cursors[1]] = 0
labels[cursors[1]:cursors[2]] = 1
labels[cursors[2]:cursors[3]] = 2
labels[cursors[3]:cursors[4]] = 3
labels[cursors[4]:cursors[5]] = 4
labels[cursors[5]:cursors[6]] = 5
labels[cursors[6]:cursors[7]] = 6

# %%
## t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("paper", font_scale=1.5,
                rc={"lines.linewidth": 2.5})    

# latent = torch.cat([latent_dhcp,latent_hcp,latent_brown],dim=0)
latent = torch.cat([latent_dhcp_dhcp_train,
                    latent_dhcp_dhcp_val,
                    latent_hcp_dhcp_train,
                    latent_hcp_dhcp_val,
                    latent_brats_dhcp_train,
                    latent_brats_dhcp_val,
                    latent_brown_dhcp],dim=0)
latent = latent.flatten(start_dim=1)
latent = latent.cpu().numpy()
tsne = TSNE(n_components=2)
latent_embedded = tsne.fit_transform(latent)

# %%
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x=latent_embedded[:,0],y=latent_embedded[:,1],hue=labels,ax=ax)
lim = (latent_embedded.min()-5, latent_embedded.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(['dHCP on dHCP Model (Train)',
            'dHCP on dHCP Model (Val)',
            'HCP on dHCP Model (Train)',
            'HCP on dHCP Model (Val)',
            'BraTS on dHCP Model (Train)',
            'BraTS on dHCP Model (Val)',
            'Unseen Paediatric Images'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
fig.savefig('figures/tsne.svg',bbox_inches='tight')

# %% [markdown]
# # Generate Latents

# %%
images = []
image_tensors = []
landmarks_t1 = op.join('/media/hdd/dhcp/dhcp_hires', 'landmarks_t1.npy')
landmarks_t2 = op.join('/media/hdd/dhcp/dhcp_hires', 'landmarks_t2.npy')
landmarks_dict = {'image_t1':landmarks_t1,'image_t2':landmarks_t2}
t2_file_list = os.listdir('/media/hdd/viscent/FLYWHEEL_BROWN/Preprocessed/reg/')
t2_file_list = [os.path.join('/media/hdd/viscent/FLYWHEEL_BROWN/Preprocessed/reg/',f'%03d.nii.gz'%x) for x in range(len(t2_file_list))]
for t2_file in tqdm(t2_file_list,total=len(t2_file_list)):
    images.append(sitk.ReadImage(t2_file))
    subject = tio.Subject(image_t2=tio.ScalarImage(t2_file))
    transform_1 = tio.Compose([
        tio.transforms.HistogramStandardization(landmarks_dict),
        tio.transforms.RescaleIntensity((0., 1.)),
        tio.transforms.ToCanonical(),
        tio.transforms.Resample((1.,1.,1.)),
    ])
    subject = transform_1(subject)
    shape = subject.image_t2.data.numpy()[0].shape
    lb = np.percentile(subject.image_t2.data.numpy(),1)
    bbox = get_bbox(subject.image_t2.data.numpy()[0],lb)
    transform_crop = tio.transforms.Crop((bbox[0], shape[0]-bbox[1], bbox[2], shape[1]-bbox[3], bbox[4], shape[2]-bbox[5]))
    subject = transform_crop(subject)
    edge_max = max(subject.image_t2.data.shape)
    padding = ((edge_max - subject.image_t2.data.shape[1]) // 2, 
                (edge_max - subject.image_t2.data.shape[2]) // 2,
                    (edge_max - subject.image_t2.data.shape[3]) // 2)
    transform_2 = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resize((160,160,160)),
    ])
    subject = transform_2(subject)
    image_tensor = subject.image_t2.data.unsqueeze(0).float()
    image_tensor = torch.transpose(image_tensor,4,2).flip([4,3,2])
    image_tensors.append(image_tensor)

# %%
display_multiplanar_center(image_tensors[9][0,0])
display_multiplanar_center(dhcp_test_loader.dataset.__getitem__(12)[1][0])
display_multiplanar_center(hcp_test_loader.dataset.__getitem__(12)[1][0])
display_multiplanar_center(brats_test_loader.dataset.__getitem__(12)[1][0])

# %% [markdown]
# ## dHCP Model

# %%
latent_dhcp = []
for img,gt in tqdm(dhcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_dhcp,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_dhcp_train.pt')

latent_hcp = []
for img,gt in tqdm(hcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_dhcp,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_dhcp_train.pt')

latent_brats = []
for img,gt in tqdm(brats_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_dhcp,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_dhcp_train.pt')

latent_dhcp = []
for img,gt in tqdm(dhcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_dhcp,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_dhcp_val.pt')

latent_hcp = []
for img,gt in tqdm(hcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_dhcp,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_dhcp_val.pt')

latent_brats = []
for img,gt in tqdm(brats_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_dhcp,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_dhcp_val.pt')

latent_brown = []
for img in tqdm(image_tensors):
    img = img.cuda()
    with torch.no_grad():
        latent_brown.append(kl_forward_latent(model_dhcp,img))
latent_brown = torch.cat(latent_brown,dim=0)
torch.save(latent_brown,'/media/hdd/viscent/SR-UNet/dev/latent_brown_dhcp.pt')

# %% [markdown]
# ## HCP Model

# %%
latent_dhcp = []
for img,gt in tqdm(dhcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_hcp,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_hcp_train.pt')
latent_hcp = []
for img,gt in tqdm(hcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_hcp,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_hcp_train.pt')
latent_brats = []
for img,gt in tqdm(brats_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_hcp,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_hcp_train.pt')
latent_dhcp = []
for img,gt in tqdm(dhcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_hcp,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_hcp_val.pt')
latent_hcp = []
for img,gt in tqdm(hcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_hcp,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_hcp_val.pt')
latent_brats = []
for img,gt in tqdm(brats_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_hcp,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_hcp_val.pt')
latent_brown = []
for img in tqdm(image_tensors):
    img = img.cuda()
    with torch.no_grad():
        latent_brown.append(kl_forward_latent(model_hcp,img))
latent_brown = torch.cat(latent_brown,dim=0)
torch.save(latent_brown,'/media/hdd/viscent/SR-UNet/dev/latent_brown_hcp.pt')

# %% [markdown]
# ## BraTS Model

# %%
latent_dhcp = []
for img,gt in tqdm(dhcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_brats,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_brats_train.pt')
latent_hcp = []
for img,gt in tqdm(hcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_brats,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_brats_train.pt')
latent_brats = []
for img,gt in tqdm(brats_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_brats,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_brats_train.pt')
latent_dhcp = []
for img,gt in tqdm(dhcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model_brats,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'/media/hdd/viscent/SR-UNet/dev/latent_dhcp_brats_val.pt')
latent_hcp = []
for img,gt in tqdm(hcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model_brats,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'/media/hdd/viscent/SR-UNet/dev/latent_hcp_brats_val.pt')
latent_brats = []
for img,gt in tqdm(brats_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_brats.append(kl_forward_latent(model_brats,img))
latent_brats = torch.cat(latent_brats,dim=0)
torch.save(latent_brats,'/media/hdd/viscent/SR-UNet/dev/latent_brats_brats_val.pt')
latent_brown = []
for img in tqdm(image_tensors):
    img = img.cuda()
    with torch.no_grad():
        latent_brown.append(kl_forward_latent(model_brats,img))
latent_brown = torch.cat(latent_brown,dim=0)
torch.save(latent_brown,'/media/hdd/viscent/SR-UNet/dev/latent_brown_brats.pt')


# %% [markdown]

# %% [markdown]
# # Quantitative Evaluation

# %%
from scipy.stats import ttest_ind
from scipy.stats import bartlett
import seaborn as sns

# %% [markdown]
# # Test on Validation dataset

# %%
models = [model_brats,model_hcp,model_dhcp]
model_picked_hcp = np.zeros(3)
divergences_hcp_per = np.zeros((len(hcp_val_loader),3))
model_correct_hcp = 0
idx = 0
for img,_ in tqdm(hcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        _,divergences,model_idx = auto_inference(models, img, latents, divergences_innate,True)
        divergences_hcp_per[idx] += divergences
        idx += 1
        model_picked_hcp[model_idx] += 1
        if model_idx == 1:  model_correct_hcp += 1
idx=0
model_picked_dhcp = np.zeros(3)
divergences_dhcp_per = np.zeros((len(dhcp_val_loader),3))
model_correct_dhcp = 0
for img,_ in tqdm(dhcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        _,divergences,model_idx = auto_inference(models, img, latents, divergences_innate,True)
        model_picked_dhcp[model_idx] += 1
        divergences_dhcp_per[idx] += divergences
        idx += 1
        if model_idx == 2:  model_correct_dhcp += 1
idx=0
model_picked_brats = np.zeros(3)
divergences_brats_per = np.zeros((len(brats_val_loader),3))
model_correct_brats = 0
for img,_ in tqdm(brats_val_loader):
    img = img.cuda()
    with torch.no_grad():
        _,divergences,model_idx = auto_inference(models, img, latents, divergences_innate,True)
        model_picked_brats[model_idx] += 1
        divergences_brats_per[idx] += divergences
        idx += 1
        if model_idx == 0:  model_correct_brats += 1


# %%
np.savetxt('data/divergences_hcp_per.csv', divergences_hcp_per, delimiter=',')
np.savetxt('data/divergences_dhcp_per.csv', divergences_dhcp_per, delimiter=',')
np.savetxt('data/divergences_brats_per.csv', divergences_brats_per, delimiter=',')
np.savetxt('data/model_picked_hcp.csv', model_picked_hcp, delimiter=',')
np.savetxt('data/model_picked_dhcp.csv', model_picked_dhcp, delimiter=',')
np.savetxt('data/model_picked_brats.csv', model_picked_brats, delimiter=',')

# %%
sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],divergences_dhcp_per.mean(axis=0))
plt.title('Wasserstein Distance of dHCP images on each model')
plt.savefig('figures/divergence_dhcp.svg',bbox_inches='tight')

sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],divergences_hcp_per.mean(axis=0))
plt.title('Wasserstein Distance of HCP images on each model')
plt.savefig('figures/divergence_hcp.svg',bbox_inches='tight')

sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],divergences_brats_per.mean(axis=0))
plt.title('Wasserstein Distance of BraTS images on each model')
plt.savefig('figures/divergence_brats.svg',bbox_inches='tight')

# %%
sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],model_picked_hcp)
plt.title('Number of called models on HCP images')
plt.savefig('figures/model_picked_hcp.svg',bbox_inches='tight')

sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],model_picked_dhcp)
plt.title('Number of called models on dHCP images')
plt.savefig('figures/model_picked_dhcp.svg',bbox_inches='tight')

sns.set_style('darkgrid')
sns.plotting_context('paper')
plt.figure(figsize=(5,7))
plt.bar(['BraTS','HCP','dHCP'],model_picked_brats)
plt.title('Number of called models on BraTS images')
plt.savefig('figures/model_picked_brats.svg',bbox_inches='tight')

# %% [markdown]
# ## dHCP-dHCP

# %%
sinkhorn_dhcp_dhcp = []
for img,gt in tqdm(dhcp_val_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_input = kl_forward_latent(model_dhcp,img)
        latent_input = latent_input.flatten().unsqueeze(0)
        latent_train = latent_dhcp.flatten(start_dim=1)
        loss = SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latent_input, latent_train)
        sinkhorn_dhcp_dhcp.append(loss.item())
sinkhorn_dhcp_dhcp = np.array(sinkhorn_dhcp_dhcp)


# %% [markdown]
# # Inference Example

# %%
plt.rcParams['figure.figsize'] = [10, 10]

# %%
torch.cuda.empty_cache()

# %% [markdown]
# ### dHCP Model

# %%
img,gt = dhcp_val_loader.dataset.__getitem__(0)
img = torch.from_numpy(img).unsqueeze(0).float().cuda()
gt = torch.from_numpy(gt).unsqueeze(0).float().cuda()
with torch.no_grad():
    pred = model_dhcp(img)
    pred_bad = model_hcp(img)
display_multiplanar_center(img[0,0].detach().cpu().numpy())
display_multiplanar_center(pred_bad[0,0].detach().cpu().numpy())
display_multiplanar_center(pred[0,0].detach().cpu().numpy())
display_multiplanar_center(gt[0,0].detach().cpu().numpy())
im_array = img[0,0].detach().cpu().numpy()
im_pred_array = pred[0,0].detach().cpu().numpy()
im_gt_array = gt[0,0].detach().cpu().numpy()
im_pred_bad_array = pred_bad[0,0].detach().cpu().numpy()
# Normalize with lowest 1% and highest value
im_array = (im_array - np.percentile(im_array,1))/(im_array.max()-np.percentile(im_array,1))
im_pred_array = (im_pred_array - np.percentile(im_pred_array,1))/(im_pred_array.max()-np.percentile(im_pred_array,1))
im_gt_array = (im_gt_array - np.percentile(im_gt_array,1))/(im_gt_array.max()-np.percentile(im_gt_array,1))
im_pred_bad_array = (im_pred_bad_array - np.percentile(im_pred_bad_array,1))/(im_pred_bad_array.max()-np.percentile(im_pred_bad_array,1))
im_sag = Image.fromarray(im_array[im_array.shape[0]//2,:,:]*255).convert('L')
im_cor = Image.fromarray(im_array[:,im_array.shape[1]//2,:]*255).convert('L')
im_ax = Image.fromarray(im_array[:,:,im_array.shape[2]//2]*255).convert('L')
im_pred_sag = Image.fromarray(im_pred_array[im_pred_array.shape[0]//2,:,:]*255).convert('L')
im_pred_cor = Image.fromarray(im_pred_array[:,im_pred_array.shape[1]//2,:]*255).convert('L')
im_pred_ax = Image.fromarray(im_pred_array[:,:,im_pred_array.shape[2]//2]*255).convert('L')
im_gt_sag = Image.fromarray(im_gt_array[im_gt_array.shape[0]//2,:,:]*255).convert('L')
im_gt_cor = Image.fromarray(im_gt_array[:,im_gt_array.shape[1]//2,:]*255).convert('L')
im_gt_ax = Image.fromarray(im_gt_array[:,:,im_gt_array.shape[2]//2]*255).convert('L')
im_pred_bad_sag = Image.fromarray(im_pred_bad_array[im_pred_bad_array.shape[0]//2,:,:]*255).convert('L')
im_pred_bad_cor = Image.fromarray(im_pred_bad_array[:,im_pred_bad_array.shape[1]//2,:]*255).convert('L')
im_pred_bad_ax = Image.fromarray(im_pred_bad_array[:,:,im_pred_bad_array.shape[2]//2]*255).convert('L')
im_sag.save('figures/outputs/dHCP_Model/sag.png')
im_cor.save('figures/outputs/dHCP_Model/cor.png')
im_ax.save('figures/outputs/dHCP_Model/ax.png')
im_pred_sag.save('figures/outputs/dHCP_Model/pred_sag.png')
im_pred_cor.save('figures/outputs/dHCP_Model/pred_cor.png')
im_pred_ax.save('figures/outputs/dHCP_Model/pred_ax.png')
im_gt_sag.save('figures/outputs/dHCP_Model/gt_sag.png')
im_gt_cor.save('figures/outputs/dHCP_Model/gt_cor.png')
im_gt_ax.save('figures/outputs/dHCP_Model/gt_ax.png')
im_pred_bad_sag.save('figures/outputs/dHCP_Model/pred_bad_sag.png')
im_pred_bad_cor.save('figures/outputs/dHCP_Model/pred_bad_cor.png')
im_pred_bad_ax.save('figures/outputs/dHCP_Model/pred_bad_ax.png')

# %% [markdown]
# ### HCP Model

# %%
img,gt = hcp_val_loader.dataset.__getitem__(3)
img = torch.from_numpy(img).unsqueeze(0).float().cuda()
gt = torch.from_numpy(gt).unsqueeze(0).float().cuda()
with torch.no_grad():
    pred = model_hcp(img)
    pred_bad = model_brats(img)
display_multiplanar_center(img[0,0].detach().cpu().numpy())
display_multiplanar_center(pred_bad[0,0].detach().cpu().numpy())
display_multiplanar_center(pred[0,0].detach().cpu().numpy())
display_multiplanar_center(gt[0,0].detach().cpu().numpy())
im_pred_array = pred[0,0].detach().cpu().numpy()
im_gt_array = gt[0,0].detach().cpu().numpy()
im_pred_bad_array = pred_bad[0,0].detach().cpu().numpy()
# Normalize with lowest 1% and highest value
im_array = (im_array - np.percentile(im_array,1))/(im_array.max()-np.percentile(im_array,1))
im_pred_array = (im_pred_array - np.percentile(im_pred_array,1))/(im_pred_array.max()-np.percentile(im_pred_array,1))
im_gt_array = (im_gt_array - np.percentile(im_gt_array,1))/(im_gt_array.max()-np.percentile(im_gt_array,1))
im_pred_bad_array = (im_pred_bad_array - np.percentile(im_pred_bad_array,1))/(im_pred_bad_array.max()-np.percentile(im_pred_bad_array,1))
im_sag = Image.fromarray(im_array[im_array.shape[0]//2,:,:]*255).convert('L')
im_cor = Image.fromarray(im_array[:,im_array.shape[1]//2,:]*255).convert('L')
im_ax = Image.fromarray(im_array[:,:,im_array.shape[2]//2]*255).convert('L')
im_pred_sag = Image.fromarray(im_pred_array[im_pred_array.shape[0]//2,:,:]*255).convert('L')
im_pred_cor = Image.fromarray(im_pred_array[:,im_pred_array.shape[1]//2,:]*255).convert('L')
im_pred_ax = Image.fromarray(im_pred_array[:,:,im_pred_array.shape[2]//2]*255).convert('L')
im_gt_sag = Image.fromarray(im_gt_array[im_gt_array.shape[0]//2,:,:]*255).convert('L')
im_gt_cor = Image.fromarray(im_gt_array[:,im_gt_array.shape[1]//2,:]*255).convert('L')
im_gt_ax = Image.fromarray(im_gt_array[:,:,im_gt_array.shape[2]//2]*255).convert('L')
im_pred_bad_sag = Image.fromarray(im_pred_bad_array[im_pred_bad_array.shape[0]//2,:,:]*255).convert('L')
im_pred_bad_cor = Image.fromarray(im_pred_bad_array[:,im_pred_bad_array.shape[1]//2,:]*255).convert('L')
im_pred_bad_ax = Image.fromarray(im_pred_bad_array[:,:,im_pred_bad_array.shape[2]//2]*255).convert('L')
im_sag.save('figures/outputs/HCP_Model/sag.png')
im_cor.save('figures/outputs/HCP_Model/cor.png')
im_ax.save('figures/outputs/HCP_Model/ax.png')
im_pred_sag.save('figures/outputs/HCP_Model/pred_sag.png')
im_pred_cor.save('figures/outputs/HCP_Model/pred_cor.png')
im_pred_ax.save('figures/outputs/HCP_Model/pred_ax.png')
im_gt_sag.save('figures/outputs/HCP_Model/gt_sag.png')
im_gt_cor.save('figures/outputs/HCP_Model/gt_cor.png')
im_gt_ax.save('figures/outputs/HCP_Model/gt_ax.png')
im_pred_bad_sag.save('figures/outputs/HCP_Model/pred_bad_sag.png')
im_pred_bad_cor.save('figures/outputs/HCP_Model/pred_bad_cor.png')
im_pred_bad_ax.save('figures/outputs/HCP_Model/pred_bad_ax.png')


# %% [markdown]
# ### BraTS Model

# %%
img,gt = brats_val_loader.dataset.__getitem__(4)
img = torch.from_numpy(img).unsqueeze(0).float().cuda()
gt = torch.from_numpy(gt).unsqueeze(0).float().cuda()
with torch.no_grad():
    pred = model_brats(img)
    pred_bad = model_hcp(img)
    pred_worse = model_dhcp(img)
display_multiplanar_center(img[0,0].detach().cpu().numpy())
display_multiplanar_center(pred_worse[0,0].detach().cpu().numpy())
display_multiplanar_center(pred_bad[0,0].detach().cpu().numpy())
display_multiplanar_center(pred[0,0].detach().cpu().numpy())
display_multiplanar_center(gt[0,0].detach().cpu().numpy())
im_array = img[0,0].detach().cpu().numpy()
im_pred_array = pred[0,0].detach().cpu().numpy()
im_gt_array = gt[0,0].detach().cpu().numpy()
im_pred_bad_array = pred_bad[0,0].detach().cpu().numpy()
im_pred_worse_array = pred_worse[0,0].detach().cpu().numpy()
# Normalize with lowest 1% and highest value
im_array = (im_array - np.percentile(im_array,1))/(im_array.max()-np.percentile(im_array,1))
im_pred_array = (im_pred_array - np.percentile(im_pred_array,1))/(im_pred_array.max()-np.percentile(im_pred_array,1))
im_gt_array = (im_gt_array - np.percentile(im_gt_array,1))/(im_gt_array.max()-np.percentile(im_gt_array,1))
im_pred_bad_array = (im_pred_bad_array - np.percentile(im_pred_bad_array,1))/(im_pred_bad_array.max()-np.percentile(im_pred_bad_array,1))
im_sag = Image.fromarray(im_array[im_array.shape[0]//2,:,:]*255).convert('L')
im_cor = Image.fromarray(im_array[:,im_array.shape[1]//2,:]*255).convert('L')
im_ax = Image.fromarray(im_array[:,:,im_array.shape[2]//2]*255).convert('L')
im_pred_sag = Image.fromarray(im_pred_array[im_pred_array.shape[0]//2,:,:]*255).convert('L')
im_pred_cor = Image.fromarray(im_pred_array[:,im_pred_array.shape[1]//2,:]*255).convert('L')
im_pred_ax = Image.fromarray(im_pred_array[:,:,im_pred_array.shape[2]//2]*255).convert('L')
im_gt_sag = Image.fromarray(im_gt_array[im_gt_array.shape[0]//2,:,:]*255).convert('L')
im_gt_cor = Image.fromarray(im_gt_array[:,im_gt_array.shape[1]//2,:]*255).convert('L')
im_gt_ax = Image.fromarray(im_gt_array[:,:,im_gt_array.shape[2]//2]*255).convert('L')
im_pred_bad_sag = Image.fromarray(im_pred_bad_array[im_pred_bad_array.shape[0]//2,:,:]*255).convert('L')
im_pred_bad_cor = Image.fromarray(im_pred_bad_array[:,im_pred_bad_array.shape[1]//2,:]*255).convert('L')
im_pred_bad_ax = Image.fromarray(im_pred_bad_array[:,:,im_pred_bad_array.shape[2]//2]*255).convert('L')
im_pred_worse_sag = Image.fromarray(im_pred_worse_array[im_pred_worse_array.shape[0]//2,:,:]*255).convert('L')
im_pred_worse_cor = Image.fromarray(im_pred_worse_array[:,im_pred_worse_array.shape[1]//2,:]*255).convert('L')
im_pred_worse_ax = Image.fromarray(im_pred_worse_array[:,:,im_pred_worse_array.shape[2]//2]*255).convert('L')
im_sag.save('figures/outputs/BraTS_Model/sag.png')
im_cor.save('figures/outputs/BraTS_Model/cor.png')
im_ax.save('figures/outputs/BraTS_Model/ax.png')
im_pred_sag.save('figures/outputs/BraTS_Model/pred_sag.png')
im_pred_cor.save('figures/outputs/BraTS_Model/pred_cor.png')
im_pred_ax.save('figures/outputs/BraTS_Model/pred_ax.png')
im_gt_sag.save('figures/outputs/BraTS_Model/gt_sag.png')
im_gt_cor.save('figures/outputs/BraTS_Model/gt_cor.png')
im_gt_ax.save('figures/outputs/BraTS_Model/gt_ax.png')
im_pred_bad_sag.save('figures/outputs/BraTS_Model/pred_bad_sag.png')
im_pred_bad_cor.save('figures/outputs/BraTS_Model/pred_bad_cor.png')
im_pred_bad_ax.save('figures/outputs/BraTS_Model/pred_bad_ax.png')
im_pred_worse_sag.save('figures/outputs/BraTS_Model/pred_worse_sag.png')
im_pred_worse_cor.save('figures/outputs/BraTS_Model/pred_worse_cor.png')
im_pred_worse_ax.save('figures/outputs/BraTS_Model/pred_worse_ax.png')
