# %% [markdown]
# 
# # Initialize

# %%
import sys
sys.path.append("/media/hdd/viscent/SR-UNet")
from utils.inference_utils import *

# %%
dhcp_train_loader, dhcp_test_loader, dhcp_val_loader = get_dataloader('dhcp',100,modality='t2')
hcp_train_loader, hcp_test_loader, hcp_val_loader = get_dataloader('hcp',100,modality='t2')


# %%
data_root = '/media/hdd/viscent/FLYWHEEL_BROWN/BROWN/SUBJECTS'
subject_list = os.listdir(data_root)
# Exclude hidden files
subject_list = [x for x in subject_list if not x.startswith('.')]
t2_file_list = []
for subject in subject_list:
    tmp = os.listdir(os.path.join(data_root, subject, 'SESSIONS'))
    tmp = [x for x in tmp if not x.startswith('.')]
    session = tmp[0]
    tmp = os.listdir(os.path.join(data_root, subject, 'SESSIONS', session, 'ACQUISITIONS'))
    tmp = [x for x in tmp if x.find('T2') != -1 and x.find('AXI') != -1 and not x.startswith('.')]
    if len(tmp) == 0:
        continue
    nifti_dir = os.path.join(data_root, subject, 'SESSIONS', session, 'ACQUISITIONS', tmp[0], 'FILES')
    nifti_file = os.listdir(nifti_dir)
    nifti_file = [x for x in nifti_file if x.endswith('.nii.gz') and not x.startswith('.')][0]
    t2_file_list.append(os.path.join(nifti_dir, nifti_file))


# %%

dhcp_t2_190 = load_model('/media/hdd/viscent/SR-UNet/pretrained_models/unet_dhcp_t2/1600.pth')

dhcp_t2_190.eval()
dhcp_t2_190.cuda()

# %% [markdown]
# ## Preprocessing

# %%
import shutil

for i,t2_file in tqdm(enumerate(t2_file_list),total=len(t2_file_list)):
    shutil.copy(t2_file, '/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_raw/%03d.nii.gz'%i)
    os.system('bet2 /media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_raw/%03d.nii.gz /media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_brain/%03d.nii.gz'% (i,i))

# %%
for i in range(113):
    img = sitk.ReadImage('/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_brain/%03d.nii.gz'%i)
    img = sitk.GetArrayFromImage(img)
    img = np.transpose(img, (1,0,2))
    sitk.WriteImage(sitk.GetImageFromArray(img), '/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_brain/%03d_.nii.gz'%i)

# %%
for i in range(113):
    os.system('flirt -searchrx -15 15 -searchry -15 15 -searchrz -15 15 -in /media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_brain/%03d.nii.gz -ref /media/hdd/viscent/SR-UNet/inference/Template_dHCP.nii.gz -out /media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_reg/%03d.nii.gz'% (i,i))

# %%
for i in range(113):
    img = sitk.ReadImage('/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/input_reg/%03d.nii.gz'%i)
    img = sitk.GetArrayFromImage(img)
    # img = np.transpose(img, (1,0,2))
    display_multiplanar_center(img)

# %%
image_tensor.shape

# %% [markdown]
# ## Sanity Check on Training Data

# %%
def kl_forward_latent(self, x):
    with torch.no_grad():
        # encoder part
        for encoder in self.encoders:
            x = encoder(x)
        return x

# %%
latents_train = []
for image_tensor, target_tensor in dhcp_train_loader:
    image_tensor = image_tensor.cuda()
    target_tensor = target_tensor.cuda()
    output_tensor = dhcp_t2_190(image_tensor).detach().cpu().numpy()
    latents_train.append(kl_forward_latent(dhcp_t2_190, image_tensor).detach().cpu().numpy())
    display_multiplanar_center(image_tensor.detach().cpu().numpy()[0,0])
    display_multiplanar_center(output_tensor[0,0])
    display_multiplanar_center(target_tensor.detach().cpu().numpy()[0,0])

# %%
latents_val = []
for image_tensor, target_tensor in dhcp_val_loader:
    image_tensor = image_tensor.cuda()
    target_tensor = target_tensor.cuda()
    output_tensor = dhcp_t2_190(image_tensor).detach().cpu().numpy()
    latents_val.append(kl_forward_latent(dhcp_t2_190, image_tensor).detach().cpu().numpy())
    display_multiplanar_center(image_tensor.detach().cpu().numpy()[0,0])
    display_multiplanar_center(output_tensor[0,0])
    display_multiplanar_center(target_tensor.detach().cpu().numpy()[0,0])

# %%
latents_hcp = []
for image_tensor, target_tensor in hcp_val_loader:
    image_tensor = image_tensor.cuda()
    target_tensor = target_tensor.cuda()
    output_tensor = dhcp_t2_190(image_tensor).detach().cpu().numpy()
    latents_hcp.append(kl_forward_latent(dhcp_t2_190, image_tensor).detach().cpu().numpy())

# %%
latents_val = torch.tensor(latents_val)
latents_train=torch.tensor(latents_train)
latents_hcp = torch.tensor(latents_hcp)

# %%
latents_train_ = latents_train.flatten(start_dim=1)
latents_val_ = latents_val.flatten(start_dim=1)
latents_hcp_ = latents_hcp.flatten(start_dim=1)

# %%
latents_val_[].shape

# %%
from geomloss import SamplesLoss
print(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_val_[2].unsqueeze(0)))
print(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_hcp_[2].unsqueeze(0)))

# %%
losses_val = []
losses_dhcp = []
for i in range(20):
    print(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_val_[i].unsqueeze(0)))
    losses_val.append(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_val_[i].unsqueeze(0)))
print('-----------------')
for i in range(20):
    print(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_hcp_[i].unsqueeze(0)))
    losses_dhcp.append(SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.95)(latents_train_, latents_hcp_[i].unsqueeze(0)))

from scipy.stats import ttest_ind

print('t-test for val and dhcp: ', ttest_ind(losses_val, losses_dhcp))

for i in range(20):
    print(losses_dhcp[i]-losses_val[i])

# %%
plt.Figure()
plt.scatter(np.arange(latents_train.shape[0]),latents_train[:,0].mean((1,2,3,4)), c='r')
# plt.scatter(latents_val[:,0,1,0,0,0], np.arange(latents_val.shape[0]), c='b')
plt.scatter(np.arange(latents_hcp.shape[0]), latents_hcp[:,0].mean((1,2,3,4)), c='g')

# %% [markdown]
# ## Testing on FLYWHEEL_BROWN data

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
    image_tensor.transpose_(2,4)
    image_tensors.append(image_tensor)
    # display_multiplanar_center(image_tensor.squeeze().squeeze().numpy())

# %%
import os


output_tensors_dhcp_190 = []

data_root = '/media/hdd/viscent/FLYWHEEL_BROWN/'

outdir_dhcp_190 = os.path.join(data_root, 'output_dhcp_1600')
# os.mkdir(outdir_dhcp_1500)
# os.mkdir(outdir_dhcp_1500_aug)
# os.mkdir(outdir_hcp_140_aug)

i=0
for image_tensor in tqdm(image_tensors):
    print('#################{}#################'.format(i))
    output_tensor_dhcp_190 = dhcp_t2_190(image_tensor.cuda()).cpu().detach()
    output_tensors_dhcp_190.append(output_tensor_dhcp_190)

    display_multiplanar_center(image_tensor.cpu().numpy()[0,0])
    display_multiplanar_center(output_tensor_dhcp_190[0,0])
    # display_multiplanar_center(output_tensor_dhcp_1500_aug[0,0])
    # display_multiplanar_center(output_tensor_hcp_140_aug[0,0])

    sitk.WriteImage(sitk.GetImageFromArray(output_tensor_dhcp_190[0,0].numpy()), os.path.join(outdir_dhcp_190, 'output.nii.gz'))
    print('####################################')
    i+=1

# %%
image_tensor = sitk.ReadImage("/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/brainex_spm.nii")

# %%
subject = tio.Subject(t2=tio.ScalarImage("/media/hdd/viscent/SR-UNet/inference/Inference_FLYWHEEL_BROWN/brainex_spm.nii"))
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
image_tensor.transpose_(2,4)
display_multiplanar_center(image_tensor)

# %%
image_tensor.shape

# %%
display_multiplanar_center(image_tensor[0,0].transpose(2,0))


# %%
image_tensor = image_tensor[0,0].transpose(2,0).unsqueeze(0).unsqueeze(0).cuda()
output_tensor = dhcp_t2_190(image_tensor)
output_array = output_tensor.cpu().detach().numpy().squeeze(0).squeeze(0)

# %%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

# %%
len(iter(dhcp_train_loader).__next__()


# %%
plt.imshow(iter(dhcp_train_loader).__next__()[0][0,0,80], cmap='gray')

# %%

plt.imshow(image_tensor[0,0,80].cpu().detach(), cmap='gray')


