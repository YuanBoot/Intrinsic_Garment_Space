import os
import glob
from common import obj_operation as obj


# Directory 
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA = os.path.join(BASE, 'data_set')
SUPPORT = os.path.join(BASE, 'support')
NNET = os.path.join(BASE, 'nnet')
CASE = os.path.join(DATA, 'case')
ANIM = os.path.join(DATA, 'anim')
GARMENT = os.path.join(DATA, 'garment')
SKELETON = os.path.join(DATA, 'skeleton')

BASIS = os.path.join(NNET, 'basis')
MIE = os.path.join(NNET, 'mie')
INFO_BASIS = os.path.join(SUPPORT, 'info_basis')
INFO_MIE = os.path.join(SUPPORT, 'info_mie')
EVAL_BASIS = os.path.join(SUPPORT, 'eval_basis')
EVAL_MIE = os.path.join(SUPPORT, 'eval_mie')
#EVAL_BASIS_LAP = os.path.join(EVAL_BASIS, 'lap_recon')
DATA_SAMPLE = os.path.join(INFO_BASIS, 'data_sample')

# File Name
CLOTH_TEMPLATE = 'cloth_template.obj'
TRAIN_LIST = 'train_list.txt'
EVAL_LIST = 'eval_list.txt'
DATA_MEAN = 'data_mean.npy'
DATA_STD = 'data_std.npy'
BASIS_MEAN = 'basis_mean.npy'
BASIS_STD = 'basis_std.npy'
MOTION_MEAN = 'motion_mean.npy'
MOTION_STD = 'motion_std.npy'

#NET = os.path.basename(glob.glob(os.path.join(BASIS, '*.net'))[0]) # just for only one .net in the folder
#TRAIN_LOG = os.path.basename(glob.glob(os.path.join(BASIS, '*.txt'))[0]) # just for only one .txt in the folder

#Data Set
CASE_01 = 'case_01'

#Charactor
JOINT_LIST = [0,1,2,3,6,7,120,122,125,128,155,157,160,163,191,192,193,194,195,196,197,198]
JOINT_NUM = len(JOINT_LIST)

# Net Parameters
tmp_verts, tmp_faces = obj.objimport(os.path.join(GARMENT, CLOTH_TEMPLATE))
VERTS_NUM = len(tmp_verts)
MESH_VERTS = tmp_verts
MESH_FACES = tmp_faces

# Basis
TRAIN_DIM = [VERTS_NUM*3, 1800, 500, 120, 30]
DP = 0.02
WEIGHT_LAP = 1.0
LEARNING_RATE = 1e-4
BATCH_EVAL = 200
BATCH_SEQ = [40,100,100,200,400,400,800,800,1600,1600]

# Motion
PRE_FRAME = 20
MOTION_DIM = [PRE_FRAME*JOINT_NUM*3, 480, 120, 30]

# latent
MIE_LERNING_RATE = 1e-3
MIE_BATCHSIZE_EVAL = 20
MIE_BATCHSIZE_01 = [4,8,8,8,16,16,16,16]
MIE_BATCHSIZE_02 = [4,4,4,8,10,10,10,10]

MIE_DIM = [30, 120, 120]
MIE_DP = 0.02

WS = [1000., 1.]


# Charactor Animation Info (one case only)
JOINTS_ANIM = os.path.basename(glob.glob(os.path.join(ANIM, 'anim_01', '*.npy'))[0])
JOINTS_ANIM_PATH = os.path.join(ANIM, 'anim_01', JOINTS_ANIM)


# File name and Path
DATA_MEAN_PATH = os.path.join(INFO_BASIS, DATA_MEAN)
DATA_STD_PATH = os.path.join(INFO_BASIS, DATA_STD)

BASIS_MEAN_PATH = os.path.join(INFO_MIE, BASIS_MEAN)
BASIS_STD_PATH = os.path.join(INFO_MIE, BASIS_STD)

MOTION_MEAN_PATH = os.path.join(INFO_MIE, MOTION_MEAN)
MOTION_STD_PATH = os.path.join(INFO_MIE, MOTION_STD)

TRAIN_LIST_PATH = os.path.join(INFO_BASIS, TRAIN_LIST)
EVAL_LIST_PATH = os.path.join(INFO_BASIS, EVAL_LIST)

CLOTH_TEMPLATE_PATH = os.path.join(GARMENT, CLOTH_TEMPLATE)


def main():
	print(BASE)


if __name__ == '__main__':
    main()