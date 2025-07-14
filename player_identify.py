import cv2
import torch
import timm
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

VIDEO_PATH  = r"C:\Users\shaur\Downloads\15sec_input_720p.mp4"
MODEL_PATH  = r"C:\Users\shaur\Downloads\best.pt"
OUTPUT_PATH = r"C:\Users\shaur\Downloads\output_cluster_robust.mp4"

# —————————————— CONFIG ——————————————
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.4
RESIZE_TO = (640, 360)
MIN_AREA = 20*20
MAX_AGE = 30
N_INIT = 3

# cost weights
W_IOU = 0.3
W_APP = 0.5
W_DIST = 0.2

print("Using device:", DEVICE)

# —————————————— YOLO  ——————————————
yolo = YOLO(MODEL_PATH).to(DEVICE)
yolo.conf = CONF_THRESH
player_cls = next(i for i,n in yolo.names.items() if n.lower()=="player")

# —————————————— Appearance Encoder ——————————————
vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).to(DEVICE)
vit.eval()
D_MODEL = vit.num_features

# —————————————— Utilities ——————————————
def compute_iou(bb, tracks_bb):
    xA = np.maximum(bb[0], tracks_bb[:,0])
    yA = np.maximum(bb[1], tracks_bb[:,1])
    xB = np.minimum(bb[2], tracks_bb[:,2])
    yB = np.minimum(bb[3], tracks_bb[:,3])
    inter = np.maximum(0, xB-xA)*np.maximum(0, yB-yA)
    areaA = (bb[2]-bb[0])*(bb[3]-bb[1])
    areaB = (tracks_bb[:,2]-tracks_bb[:,0])*(tracks_bb[:,3]-tracks_bb[:,1])
    return inter/(areaA+areaB-inter+1e-6)

def appearance_matrix(curr_feats, tracks):
    N, M = curr_feats.shape[0], len(tracks)
    if M==0:
        return np.zeros((N,0), dtype=float)
    track_feats = torch.stack([t.feat for t in tracks], dim=0).to(DEVICE)
    det_norm    = curr_feats / curr_feats.norm(dim=1,keepdim=True)
    track_norm  = track_feats / track_feats.norm(dim=1,keepdim=True)
    sim = (det_norm @ track_norm.T).detach().cpu().numpy()
    return np.clip(sim, 0, 1)

def center_distance_matrix(dets, tracks, diag):
    # normalized Euclidean distance between detection and track centers
    N, M = len(dets), len(tracks)
    mat = np.zeros((N, M), dtype=float)
    if M==0:
        return mat
    det_centers = np.array([[(d[0]+d[2])/2, (d[1]+d[3])/2] for d in dets])
    track_centers = np.array([t.get_center() for t in tracks])
    for i, dc in enumerate(det_centers):
        dists = np.linalg.norm(track_centers - dc, axis=1)
        mat[i,:] = dists/diag
    return np.clip(mat, 0, 1)

def fused_cost(iou, app, dist):
    return W_IOU*(1-iou) + W_APP*(1-app) + W_DIST*dist

class Track:
    def __init__(self, tid, bbox, feat):
        self.id = tid
        self.hits = 1
        self.time_since_update = 0
        
        # Kalman for [cx,cy,w,h, vx,vy,vw,vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt=1.
        F=np.eye(8)
        for i in range(4): F[i,i+4]=dt
        self.kf.F, self.kf.H = F, np.zeros((4,8))
        for i in range(4): self.kf.H[i,i]=1
        self.kf.P*=10
        self.kf.R*=1
        self.kf.Q*=0.01
        x1,y1,x2,y2 = bbox
        w=x2-x1
        h=y2-y1
        cx=x1+w/2
        cy=y1+h/2
        self.kf.x[:4]=np.array([[cx,cy,w,h]]).T
        self.feat = feat.detach().cpu()
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        self.hits = max(self.hits,1)
    def update(self, bbox, feat):
        x1,y1,x2,y2 = bbox
        w=x2-x1
        h=y2-y1
        cx=x1+w/2
        cy=y1+h/2
        self.kf.update(np.array([cx,cy,w,h]))
        self.feat = 0.5*self.feat + 0.5*feat.detach().cpu()
        self.hits += 1
        self.time_since_update = 0
    def get_state(self):
        cx,cy,w,h = self.kf.x[:4].flatten()
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    def get_center(self):
        s = self.get_state()
        return np.array([(s[0]+s[2])/2, (s[1]+s[3])/2])

# —————————————— Main Loop ——————————————
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(3)), int(cap.get(4))
diag = np.sqrt(w*w + h*h)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

tracks, next_id = [], 0

while True:
    ret, frame = cap.read()
    if not ret: break
    proc = cv2.resize(frame, RESIZE_TO) if RESIZE_TO else frame

    # detect
    dets=[]
    for res in yolo(proc):
        for b in res.boxes:
            if int(b.cls.item())!=player_cls or float(b.conf.item())<CONF_THRESH: continue
            x1,y1,x2,y2 = map(int, b.xyxy.detach().cpu().numpy().flatten())
            if RESIZE_TO:
                sx,sy=w/RESIZE_TO[0],h/RESIZE_TO[1]
                x1,x2=int(x1*sx),int(x2*sx)
                y1,y2=int(y1*sy),int(y2*sy)
            if (x2-x1)*(y2-y1)<MIN_AREA: continue
            dets.append((x1,y1,x2,y2))

    # predict
    for tr in tracks: tr.predict()

    # features
    feats=[]
    for bb in dets:
        x1,y1,x2,y2=bb
        patch=frame[y1:y2,x1:x2].copy()
        crop=cv2.resize(patch,(224,224))
        img=np.ascontiguousarray(crop[:,:,::-1])
        inp=(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)/255.0)
        feats.append(vit(inp).squeeze(0))
    curr_feats = torch.stack(feats) if feats else torch.zeros((0,D_MODEL),device=DEVICE)

    # build cost
    num_det, num_tr = len(dets), len(tracks)
    if num_tr>0 and curr_feats.size(0)>0:
        iou_mat  = np.zeros((num_det,num_tr))
        preds    = np.stack([tr.get_state() for tr in tracks])
        for i,bb in enumerate(dets):
            iou_mat[i]=compute_iou(bb,preds)
        app_mat  = appearance_matrix(curr_feats, tracks)
        dist_mat = center_distance_matrix(dets, tracks, diag)
        cost_mat = fused_cost(iou_mat, app_mat, dist_mat)
    else:
        cost_mat = np.zeros((num_det,num_tr))

    # assign
    assigned_det,set_tr=set(),set()
    if num_tr>0 and num_det>0:
        row,col = linear_sum_assignment(cost_mat)
        for r,c in zip(row,col):
            if cost_mat[r,c]<1.0:
                tracks[c].update(dets[r], curr_feats[r])
                assigned_det.add(r)
                set_tr.add(c)

    # spawn new
    for i,bb in enumerate(dets):
        if i not in assigned_det:
            tracks.append(Track(next_id, bb, curr_feats[i]))
            next_id+=1

    # prune
    tracks=[tr for tr in tracks if tr.time_since_update<=MAX_AGE]

    # draw
    for tr in tracks:
        if tr.hits>=N_INIT:
            x1,y1,x2,y2=map(int,tr.get_state())
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID {tr.id}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    writer.write(frame)

cap.release()
writer.release()
print("Video Saved at ", OUTPUT_PATH)