import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine): # train or test, 500, True, datasets/linemod/Linemod_preprocessed, 0.03, False or True
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                # if self.mode == 'test' and item_count % 10 != 0:
                #     continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line)) # rgb image
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line)) # depth image
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line)) #segnet 결과 mask 이미지 #{2}_label.png
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line)) # mask 이미지

                self.list_obj.append(item) # 객체 번호 1,2,4,5,6,8,9,10,11,12,13,14,15
                self.list_rank.append(int(input_line)) # .txt 에서 읽은 이미지 번호
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r') # ground truth
            self.meta[item] = yaml.safe_load(meta_file)
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item)) # .ply에서 앞에 3개 포인트 클라우드 파일인듯 / 앞에 x,y,z만 가져옴
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110 #320.44989013671875
        self.cam_cy = 242.04899 #244.81730651855469
        self.cam_fx = 572.41140 #614.04742431640625
        self.cam_fy = 573.57043 #614.044677734375

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) # 세로로(y방향) 값 변함
        self.ymap = np.array([[i for i in range(640)] for j in range(480)]) # 가로로(x방향) 값 변함
        
        self.num = num #500
        self.add_noise = add_noise #True
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) # data augmentation
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500 #500 #168
        self.num_pt_mesh_small = 500 #500 #168
        self.symmetry_obj_idx = [7,8,16]

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]        
        
        if obj == 2: # data02 gt.yml에는 obj_id가 2말고 다른거도 있음
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i] # obj : 해당 gt.yml / rank : 파일 안에서 몇 번째 정보인지 / [0] : 리스트로 나옴 안하면 {[]} 이 형태
                    break
        else: # 나머지 gt.yml은 해당 obj_id만 있음
            meta = self.meta[obj][rank][0] # obj : 해당 gt.yml / rank : 파일 안에서 몇 번째 정보인지 / [0] : 리스트로 나옴 안하면 {[]} 이 형태
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # 0 아닌 부분 --(True)
        if self.mode == 'eval':
            if obj == 16:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(36))) # new_dr 102
            elif obj == 17:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(254)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            if obj == 16:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(36))) # new_dr 102
                #mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([127, 127, 127])))[:, :, 0]
            elif obj == 17:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(254)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0] # masked_equal [255,255,255]인 부분 -- (True)
        mask = mask_label * mask_depth # mask 생성(640x480)
        if self.add_noise:
            img = self.trancolor(img) # data augmentation

        img = np.array(img)[:, :, :3] # [높이][세로][가로]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img # 640x480x3으로 만들어줌

        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax] # rgb_img에 해당 바운딩 박스 부분만 mask
        # p_img = np.transpose(img_masked, (1, 2, 0))
        # cv2.imwrite('./datasets/linemod/Linemod_preprocessed/data/17/evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3)) # target object의 R matrix
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)]) # noise

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # mask img의 바운딩 박스에서 0이 아닌 인덱스 반환
        if len(choose) == 0: # mask가 없는 경우
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num: # 0이 아닌 인덱스가 500보다 큰 경우
            c_mask = np.zeros(len(choose), dtype=int) # len(choose) 만큼의 0으로 된 배열 생성
            c_mask[:self.num] = 1 # 0~499까지 1로 만듬
            np.random.shuffle(c_mask) # 섞음
            choose = choose[c_mask.nonzero()] # 0이 아닌 인덱스
        else: # 500보다 작은 경우
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        # choose로 H, W 유지시켜주는 전처리
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # masked depth image에서 포인트 클라우드 만들기 위해 역투영
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale # z
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx # x
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy # y
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0 #(x,y,z)

        if self.add_noise: # noise 추가
            cloud = np.add(cloud, add_t)

        # fw = open('./datasets/linemod/Linemod_preprocessed/data/17/evaluation_result/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # 객체 포인트 너무 많아서 랜덤으로 몇 개 지워주는 부분 
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        # fw = open('./datasets/linemod/Linemod_preprocessed/data/17/evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # 타겟 객체의 포즈 R,T 를 이용해 구해줌
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        # fw = open('./datasets/linemod/Linemod_preprocessed/data/17/evaluation_result/{0}_tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # cloud : masked 포인트 클라우드
        # choose : H, W 안 변하게 해주는 전처리
        # img_masked : color img 마스크
        # target : 타겟 객체의 pos
        # model_points : 객체 몇몇 포인트(예측 포즈 구하기 위함)
        # objlist : 해당 객체 번호 1,2,4,5,6,8,9,10,11,12,13,14,15
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
