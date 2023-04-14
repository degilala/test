import torch
import random
import numpy as np
import copy
import cv2
import time

import utils
from dataloader import BrainData
from config import Config as cfg
from brain import SPAC
from env import Env
from summary import Summary
from networks import *
import torch
import random
import numpy as np
import copy
import cv2
import time
import SimpleITK as itk
import data_util.brain
import data_util.liver
import utils
from dataloader import BrainData
from config import Config as cfg
from brain import SPAC
from env import Env
from summary import Summary
from networks import *
from torchvision import transforms
import myDataset
# from thop import profile
# from thop import clever_format
from torch.utils.data import DataLoader
# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致
def get_register_dataloder(test_dir):
    train_transform = transforms.Compose([transforms.ToTensor()])
    reg_data = myDataset.pair_Dataset(data_dir=test_dir, transform=train_transform)
    #reg_data = lung_iowa_list.LungDataset._get_file_names(self)
    reg_loader = DataLoader(dataset=reg_data, batch_size=1)
    assert len(reg_loader) > 0, 'Could not find any training data'
    return reg_data, reg_loader
if __name__ == "__main__":
    idx = 100
    setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)
    #######################################
    # stn = SpatialTransformer(device).to(device)
    stn = SpatialTransformer(cfg.HEIGHT, 'bilinear').to(device)
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device)
    #test_data = BrainData(cfg.TEST_DATA, mode='test', size=cfg.HEIGHT, affine=False)
    #test_loader = BrainData(cfg.TEST_DATA, mode='test')
    # test_loader = test_data.generator()
    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE))#根据 cfg.IMAGE_TYPE 的值来选择不同的数据集类型，并将其赋值给变量
    # Dataset，以便后续的代码可以使用该数据集类型进行数据处理
    dataset = Dataset(split_path='datasets/%s.json' % cfg.IMAGE_TYPE, paired=False,affine=False)
    test_loader = dataset.generator(cfg.DATA_TYPE, batch_size=1, loop=True)
    reg_data, reg_loader = get_register_dataloder((r'F:\peizhun\SPAC-Deformable-Registration-main\mydata\128', r'F:\peizhun\SPAC-Deformable-Registration-main\mydata\moved'))
    brain = SPAC(stn, device)
    brain.load_planner(r'F:\peizhun\SPAC-Deformable-Registration-main\code\model\planner_16000.ckpt')
    #brain.load_model('decoder', cfg.DECODER_MODEL_RL)
    brain.load_actor(r'F:\peizhun\SPAC-Deformable-Registration-main\code\model\actor_16000.ckpt')
    brain.load_critic(r'F:\peizhun\SPAC-Deformable-Registration-main\code\model\critic1_16000.ckpt',r'F:\peizhun\SPAC-Deformable-Registration-main\code\model\critic2_16000.ckpt')
    #brain.load_model('critic2_16000', cfg.CRITIC2_MODEL)

    brain.eval(brain.actor)
    brain.eval(brain.critic1)
    brain.eval(brain.critic2)
    brain.eval(brain.planner)

    #model = VAE(cfg).to(device)
    # input = torch.randn(1, 2, 128, 128, 128).cuda()
    # macs, params = profile(model, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('flops: {}, params: {}'.format(macs, params))

    dices = []
    times = []
    # grid = utils.virtual_grid(cfg.HEIGHT, tensor=True).unsqueeze(0).to(device)
    # print(grid.shape)
    print(len(reg_loader))
    latents = []
    labs = []
    jacobian_dets = []
    for i, item in enumerate(reg_loader):
        fixed = item[0]
        moving = item[1]
        #fixed = item['fixed']
        #fixed_seg = item['fixed_seg']
        #moving = item['moving']
        #moving_seg = item['moving_seg']

        #if i % 1 == 0:
        #    cv2.imwrite('{}/{}-fixed.bmp'.format(cfg.TEST_PATH, i), fixed[:, :, idx])
        #    cv2.imwrite('{}/{}-moving.bmp'.format(cfg.TEST_PATH, i), moving[:, :, idx])
            # cv2.imwrite('{}/{}-fixed_seg.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(fixed_seg>0, 255.)[:, :, idx])
            # cv2.imwrite('{}/{}-moving_seg.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(moving_seg>0, 255.)[:, :, idx])

        #fixed_seg = utils.numpy_im(fixed_seg, 1)
        #moving_seg = moving_seg.to(device)[None,...]
        fixed = fixed.to(device)
        moving = moving.to(device)

        moved = copy.deepcopy(moving)

        pred = None
        best_pred = None
        best_value = 0
        best_step = 0
        step = 0
        tic = time.time()

        state = torch.cat([fixed, moving], dim=1)
        latent, flow = brain.choose_action(state)

        latents.append(latent.cpu().numpy())
        labs.append(i)
        save_name = reg_data.data_info[i][1].split('\\')[-1]
        #pred = flow if pred is None else stn(pred, flow) + flow
        warped_im = stn(moving, flow).cpu().detach().numpy()
        toc = time.time()
        wrap_save_path=r"F:\peizhun\SPAC-Deformable-Registration-main\mydata\moved_step"+"/"+str(save_name)+str(i)+".nii"
        #warped_im = stn(moving, pred)
        fixed_path = r'F:\peizhun\SPAC-Deformable-Registration-main\mydata\128\mni_icbm152_t1_tal_nlin_sym_09a_brain.nii'
        fixed_sitk = itk.ReadImage(fixed_path)
        warp_ori = itk.GetImageFromArray(warped_im[0,0,:,:,:])
        warp_ori.SetDirection(fixed_sitk.GetDirection())
        warp_ori.SetOrigin(fixed_sitk.GetOrigin())
        warp_ori.SetSpacing(fixed_sitk.GetSpacing())
        itk.WriteImage(warp_ori, wrap_save_path)
        #warped_seg = utils.numpy_im(seg_stn(moving_seg, pred), 1, device)#.astype(np.uint8)
        # moving_seg_numpy = utils.numpy_im(moving_seg, 1, device=device)
        # warped_grid = utils.numpy_im(seg_stn(grid, pred), device=device).astype(np.uint8)

        # scores = []
        # warped_seg = cv2.blur(warped_seg, (3, 3))
        #labels = np.unique(fixed_seg)[1:]

        #scores = utils.dice(fixed_seg>0, warped_seg>0, [1])
        #score = np.mean(scores)
        #dices.append(score)

        times.append(toc-tic)


