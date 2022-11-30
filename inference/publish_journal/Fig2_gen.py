# %% [markdown]
# # Visualizing latent space with t-SNE

# %% [markdown]
# ## Initialization

# %%
import sys
sys.path.append("/media/hdd/viscent/SR-UNet")
from utils.inference_utils import *
from geomloss import SamplesLoss
from skopt import gp_minimize
import torchvision
import kornia as K

# %%
dhcp_train_loader, dhcp_test_loader, dhcp_val_loader = get_dataloader('dhcp',50,modality='t2')
hcp_train_loader, hcp_test_loader, hcp_val_loader = get_dataloader('hcp',50,modality='t2')

# %%
%%capture 
model = load_model('/media/hdd/viscent/SR-UNet/pretrained_models/unet_dhcp_t2/1600.pth')
model = model.cuda()
model.eval()

# %%
latent_dhcp = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_dhcp.pt')
latent_hcp = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_hcp.pt')
latent_brown = torch.load('/media/hdd/viscent/SR-UNet/dev/latent_brown.pt')

# %%
labels = np.zeros(latent_dhcp.shape[0]+latent_hcp.shape[0])
labels[latent_dhcp.shape[0]:] = 1

# %% [markdown]
# ## t-SNE 

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
latent = torch.cat([latent_dhcp,latent_hcp],dim=0)
latent = latent.flatten(start_dim=1)
latent = latent.cpu().numpy()
latent_embedded = TSNE(n_components=2).fit_transform(latent)

# %% [markdown]
# ## Visualizing the t-SNE output

# %%
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x=latent_embedded[:,0],y=latent_embedded[:,1],hue=labels,ax=ax)
lim = (latent_embedded.min()-5, latent_embedded.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(['dHCP','HCP'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
fig.savefig('tsne.svg',bbox_inches='tight')

# %% [markdown]
# # Generate Latents

# %%
images = []
image_tensors = []
t2_file_list = os.listdir('/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_raw/')
t2_file_list = [os.path.join('/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_raw/',x) for x in t2_file_list]
for t2_file in tqdm(t2_file_list,total=len(t2_file_list)):
    images.append(sitk.ReadImage(t2_file))
    subject = tio.Subject(t2=tio.ScalarImage(t2_file))
    transform_1 = tio.Compose([
        tio.transforms.RescaleIntensity((0., 1.)),
        tio.transforms.ToCanonical(),
        tio.transforms.Resample((1.,1.,1.)),
    ])
    subject = transform_1(subject)
    edge_max = max(subject.t2.data.shape)
    padding = ((edge_max - subject.t2.data.shape[1]) // 2, 
                (edge_max - subject.t2.data.shape[2]) // 2,
                    (edge_max - subject.t2.data.shape[3]) // 2)
    transform_2 = tio.Compose([
        tio.Pad(padding),
        tio.transforms.Resize((160,160,160)),
    ])
    subject = transform_2(subject)
    image_tensor = subject.t2.data.unsqueeze(0).float()
    image_tensors.append(image_tensor)

# %%
latent_dhcp = []
for img,gt in tqdm(dhcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_dhcp.append(kl_forward_latent(model,img))
latent_dhcp = torch.cat(latent_dhcp,dim=0)
torch.save(latent_dhcp,'latent_dhcp.pt')
latent_hcp = []
for img,gt in tqdm(hcp_train_loader):
    img = img.cuda()
    with torch.no_grad():
        latent_hcp.append(kl_forward_latent(model,img))
latent_hcp = torch.cat(latent_hcp,dim=0)
torch.save(latent_hcp,'latent_hcp.pt')
latent_brown = []
for img in tqdm(image_tensors):
    img = img.cuda()
    with torch.no_grad():
        latent_brown.append(kl_forward_latent(model,img))
latent_brown = torch.cat(latent_brown,dim=0)
torch.save(latent_brown,'latent_brown.pt')

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
sns.plotting_context('talk')

x1 = np.random.normal(10, 5, 100)
x2 = np.random.normal(35, 5, 100)

x = np.stack((x1, x2), axis=1)

sns.displot(x, aspect=16/9, bins=12, kde=True)
plt.savefig('wasserstein_sched.svg',bbox_inches='tight')

# %%



