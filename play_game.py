import sys, argparse
sys.path.append('game/')
import flappy_wrapped as game
import cv2
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim

KERNEL = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
def processFrame(frame):
    frame = frame[55:288,0:400] #crop image
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to black and white
    frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
    _ , frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
    frame = cv2.filter2D(frame,-1,KERNEL)
    frame = frame.astype(np.float64)/255.0
    return frame

#Dueling DQN
class DDQN(nn.Module):
    def __init__(self,input_shape,nactions):
        super(DDQN,self).__init__()
        self.nactions = nactions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=2,stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fca = nn.Sequential(
            nn.Linear( conv_out_size, 512),
            nn.ReLU(),
            nn.Linear( 512, nactions )
        )
        
        self.fcv = nn.Sequential(
            nn.Linear(conv_out_size,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
        
    def _get_conv_out(self,shape):
        o = self.conv( torch.zeros(1,*shape) )
        return int(np.prod(o.size()))
    
    def forward(self,x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        action_v = self.fca(conv_out)
        value_v = self.fcv(conv_out).expand(x.size(0), self.nactions)
        return value_v + action_v - action_v.mean(1).unsqueeze(1).expand(x.size(0), self.nactions)
    
STATE_DIM = 4
SKIP_FRAME = 2
INITIAL_SKIP = [0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1]

def initial_autoplay(env):
    state = collections.deque(maxlen=STATE_DIM)
    for i in INITIAL_SKIP[:-7]:
        frame,reward,done = env.frame_step(i)
    frame = processFrame(frame)
    state.append(frame)
    
    for i in INITIAL_SKIP[-7:-5]:
        frame,reward,done = env.frame_step(i)
    frame = processFrame(frame)
    state.append(frame)
        
    for i in INITIAL_SKIP[-5:-3]:
        frame,reward,done = env.frame_step(i)
    frame = processFrame(frame)
    state.append(frame)
        
    for i in INITIAL_SKIP[-3:-1]:
        frame,reward,done = env.frame_step(i)
    frame = processFrame(frame)
    state.append(frame)
    return state
    
if __name__=='__main__':
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    
    env = game.GameState()
    args = parser.parse_args()
    net = DDQN( (STATE_DIM,84,84), 2 ).to(device)
    net.load_state_dict(torch.load('checkpoints/flappy_best_model.dat'))
    
    input("Please Press Enter to Start")
    state = initial_autoplay(env)
    total_rewards = 0
    while True:
        state_v = torch.tensor(np.array([state],copy=False),dtype=torch.float32).to(device)
        action = int(torch.argmax(net(state_v)))
        
        frame,reward,done = env.frame_step(action)
        total_rewards += reward
        for _ in range(SKIP_FRAME):
            frame,reward,done =  env.frame_step(action)
            total_rewards += reward
            if done:
                break
                
        frame = processFrame(frame)
        state.append(frame)
        
        
        