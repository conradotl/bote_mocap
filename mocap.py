import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as signal

### define os parametros a serem usados aqui
segmentos = {
    'antebraco_E':[13,15],
    'braco_E':[11,13],
    'antebraco_D':[14,16],
    'braco_D':[12,14],
    'tronco':[7,11,23,24,12,8], # incluindo a cabeca
    'coxa_E':[23,25],
    'perna_E':[25,27],
    'coxa_D':[24,26],
    'perna_D':[26,28]
}
massa_segmentos = { #/10.1016/j.jbiomech.2019.03.016
    'antebraco_E':1.5,
    'braco_E':3.3,
    'antebraco_D':1.5,
    'braco_D':3.3,
    'tronco':42.8 + 15.8, # incluindo a cabeca
    'coxa_E':11.4,
    'perna_E':4.5,
    'coxa_D':11.4,
    'perna_D':4.5
}
articulacoes = {
    'ombro_E' : [23,11,13],
    'cotovelo_E':[11,13,15],
    'quadril_E':[11,23,25],
    'joelho_E':[23,25,27],
    'ombro_D' : [24,12,14],
    'cotovelo_D':[12,14,16],
    'quadril_D':[12,24,26],
    'joelho_D':[24,26,28]
}

### define as funcoes e classes
# gerenciamento dos videos
def get_analysis_epoch(vid_path,fliped=False):
    # cria o objeto de captura do cv (video)
    cap = cv2.VideoCapture(vid_path)
    # registra dimensoes da imagem
    height,width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FPS))
    # registra o numero de frames
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # dic para receber inicio e fim do periodo de analise 
    periodo_analise = {}
    # instancia um contador para frames e para teclar presionadas
    count = 0
    event_count = 0
    # loop de exibicao
    while True:
        count += 1
        # le o proximo frame
        ret,frame = cap.read()
        # verifica se há um frame para exibir
        if not ret:
            print('Quebra de exibição. frame ',count)
            # se a falta de frame for antes do fim do video, pula o frame 
            if count < frame_count-1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, count+1)
                continue
            # senao para exibicao
            else:
                break
        # flipa a imagem se indicado
        if fliped:
            frame = cv2.flip(frame, 0)
        # redimensiona a imagem
        frame = cv2.resize(frame, ((int(width/2),int(height/2))))
        # mostra o frame
        cv2.imshow('Qualquer tecla para inicio e fim do periodo de analise. Q para sair',frame)
        # registra input
        key = cv2.waitKey(1)
        # quebra
        if key & 0xFF == ord('q'):
            break
        # atualiza o inicio e fim do periodo de analise e quebra
        elif key != -1:
            if event_count == 0:
                periodo_analise.update({'start':count})
                print('começo:',count)
            elif event_count == 1:
                periodo_analise.update({'end':count})
                print('fim:',count)
            elif event_count > 1:
                break
            event_count += 1
    # solta os obejetos e destroi a janela
    cap.release()
    cv2.destroyAllWindows()
    return periodo_analise

def get_roi(vid_path,periodo_analise,fliped=False):
    # junta todos os frames do periodo de analise
    # cria o objeto de captura do cv (video)
    cap = cv2.VideoCapture(vid_path)
    # registra dimensoes da imagem
    height,width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # passa e extrai os frames da seleção
    frames = []
    for i in range(periodo_analise['start'],periodo_analise['end'],100):
        # pula para o frame em questão
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        try:
            frame = cv2.resize(cap.read()[1], ((int(width/2),int(height/2))))
            # flipa a imagem se indicado
            if fliped:
                frame = cv2.flip(frame, 0)
            frames.append(frame)
        except Exception as error:
            print(error)

    # sobrepoe as imagens para que o usuario defiuna uma regiao de interese (ROI)
    img = np.array(frames).mean(0).astype('uint8')
    # mostra a imagem
    roi = cv2.selectROI('Selecione a regiao de analise e pressione qualquer tecla',img)
    # Aguarde até que o usuário feche a janela
    key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    return roi

class Calibrator:
    def __init__(self):
        self.points = []
        self.distance = 0
    # controla os clicks e gerenciamento dos pontos 
    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("image", self.img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("image", self.img)
            cv2.line(self.img, self.points[-2], self.points[-1], (0, 255, 0), 2)
            d = np.sqrt((self.points[-2][0] - self.points[-1][0])**2 + (self.points[-2][1] - self.points[-1][1])**2)
            self.distance = d
    
    def get_distance_pixels(self,vid_path,periodo_analise,fliped=False):
        # junta todos os frames do periodo de analise/ cria o objeto de captura do cv (video)
        cap = cv2.VideoCapture(vid_path)
        # registra dimensoes da imagem
        height,width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # passa e extrai os frames da seleção
        frames = []
        for i in range(periodo_analise['start'],periodo_analise['end'],100):
            # pula para o frame em questão
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            try:
                frame = cv2.resize(cap.read()[1], ((int(width/2),int(height/2))))
                # flipa a imagem se indicado
                if fliped:
                    frame = cv2.flip(frame, 0)
                frames.append(frame)
            except Exception as error:
                print(error)
        # sobrepoe as imagens para que o usuario defiuna uma regiao de interese (ROI)
        self.img = np.array(frames).mean(0).astype('uint8')     
        # load the image, clone it, and setup the mouse callback function
        clone = self.img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.img)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.img = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("q"):
                break
        # close all open windows
        cv2.destroyAllWindows()
        return self.distance

# processamento dos videos
def get_virtual_markers_coords_single_image(frame,pose):
    '''
    Apply predefined neural network to extract relative image coordinates of joint centers from array of images
    frames [array of arrays] : arrays of RGB images
    pose [obj] : mediapipe pose object

    return [array of arrays] : virtual markers location from each image
    '''

    # get images dimension
    height,width = frame.shape[:-1]
    # processa usando o pose
    result = pose.process(frame)
    # if sucessful, get coords
    if result.pose_landmarks:
        # extract the x and y coords and zip it
        xs=[landmark.x * width for landmark in result.pose_landmarks.landmark]
        ys=[landmark.y * height for landmark in result.pose_landmarks.landmark]
        coord = np.array(list(zip(xs,ys)))
    # if not, insert nans
    else:
        # pass nans
        coord = np.full((33,2),np.nan)
            
    return np.array(coord)

def analyze_video(vid_path,periodo_analise,roi,fliped=False):
    # instancia um avaliador do mediapipe
    pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity = 2,
                                  min_detection_confidence = .5,
                                  min_tracking_confidence = .85)
    # cria uma lista para receber as coordenadas
    coords = []
    # cria o objeto de captura do cv (video)
    cap = cv2.VideoCapture(vid_path)
    # registra dimensoes da imagem
    height,width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # pula para inicio do periodo de analise
    cap.set(cv2.CAP_PROP_POS_FRAMES, periodo_analise['start'])
    # loop de exibicao
    count = 1
    while count < (periodo_analise['end']-periodo_analise['start'])/2:
        count += 1
        # le o proximo frame
        ret,frame = cap.read()
        # verifica se há um frame para exibir
        if not ret:
            print('Quebra de exibição. frame ',count)
            # se a falta de frame for antes do fim do video, pula o frame 
            if count < frame_count-1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, count+1)
                continue
            # senao para exibicao
            else:
                break
        # redimensiona a imagem
        frame = cv2.resize(cap.read()[1], ((int(width/2),int(height/2))))
        # flipa a imagem se indicado
        if fliped:
            frame = cv2.flip(frame, 0)
        # corta a imagem
        frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        # recupera as coordenada do mediapipe
        coord = get_virtual_markers_coords_single_image(frame,pose)
        coords.append(coord)
        # imprime os marcadores virtuais nos frames
        if not np.all(np.isnan(coord)):
            # tronco
            frame = print_limb(frame,coord[[7,11,23,24,12,8]].astype(int),(255,255,255))
            # lado direito
            frame = print_limb(frame,coord[[12,14,16]].astype(int),(255,0,0))
            frame = print_limb(frame,coord[[24,26,28]].astype(int),(255,0,0))
            # lado esquerdo
            frame = print_limb(frame,coord[[11,13,15]].astype(int),(0,0,255))
            frame = print_limb(frame,coord[[23,25,27]].astype(int),(0,0,255))
            # CoM
            frame = print_limb(frame,[get_CoM(coord).astype(int)],(0,255,255),line=False)
        # mostra o frame
        cv2.imshow('Qualquer tecla para inicio e fim do periodo de analise. Q para sair',frame)
        # registra input
        key = cv2.waitKey(1)
        # quebra
        if key & 0xFF == ord('q'):
            break
    # solta os obejetos e destroi a janela
    cap.release()
    cv2.destroyAllWindows()
    # exclui os frames não detectados
    coords = np.array([coord for coord in coords if not np.all(np.isnan(coord))])
    return coords

def process_coords(coords,calib_factor,fps=120):
    # centraliza no centro de massa inicial
    coords = coords - get_CoM(coords[0])
    coords = coords * calib_factor
    # filtra
    coords = signal.filtfilt(*signal.butter(4,8,fs=fps),coords,axis=0)
    # flipa vertical
    coords[:,:,1] = -coords[:,:,1]
    return coords

def get_joint_angles(articulacoes,coords):
    angles = {}
    for art in articulacoes.keys(): 
        angles_art = [get_angle(*coord[articulacoes[art]]) for coord in coords]
        angles.update({art:angles_art})
    return angles

def find_fly_phase(coords,fps=120):
    altura_mao = (coords[:,(15,16),1]).mean(1)
    altura_mao = altura_mao - altura_mao[0]
    # filtr\
    b,a= signal.butter(4,6,fs=fps)
    acc_mao = signal.filtfilt(b,a,np.diff(altura_mao,2)*fps**2)
    # detecta os pontos de inicio e fim do movimento da mão
    above_t = np.where(np.abs(acc_mao)>np.mean(acc_mao)+np.std(acc_mao))[0]
    start, end = above_t[0],above_t[-1]
    return start,end

def get_CoM(coord,segmentos=segmentos,massa_segmentos=massa_segmentos):
    CoM = (np.array([coord[segmentos[seg]].mean(0)*massa_segmentos[seg] for seg in segmentos.keys()])).sum(0)/sum(massa_segmentos.values())
    return CoM
# funcoes utilitarias
def get_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def moving_average(arr, window_size=30):
    '''
    pass a flat moving average of a given window size. Convolution is done bidirectionally to minimize time shift 
    '''
    # make a kernel to convolution over
    weights = np.ones(window_size) / window_size
    # pass it forwardly
    mov_avg = np.convolve(arr, weights, 'valid')[:len(arr)]
    # pass it backwardly
    mov_avg = (np.convolve(mov_avg[::-1], weights, 'valid')[:len(arr)])[::-1]
    # insert side padding to equal input length
    padding = int((len(arr)-len(mov_avg))/2)
    mov_avg = np.concatenate([np.repeat(mov_avg[0],padding),mov_avg,np.repeat(mov_avg[-1],padding)])
    
    return mov_avg

def save_excel(path,coords,joint_angles,fps=120):
    # calcula centro de massa
    CoM = np.array([get_CoM(coord) for coord in coords])
    altura_CoM = CoM[:,1]
    # calcula velocidade do coentro de massa
    vel_CoM = np.diff(altura_CoM)*fps
    vel_CoM = np.insert(vel_CoM,-1,0)
    # calcula altura das mãos
    altura_mao = (coords[:,(15,16),1]).mean(1)
    altura_mao = altura_mao - altura_mao[0]
    # insere em um dataframe
    sinal_df = pd.DataFrame({
        'altura_centro_de_massa':altura_CoM,
        'velocidade_vertical_centro_de_massa':vel_CoM,
        'altura_maos':altura_mao})
    sinal_df = pd.concat([sinal_df,pd.DataFrame(joint_angles)],axis=1)
    # calcula os dados discretos
    # encontra fase de voo
    start, end = find_fly_phase(coords,fps=120)
    # calcula as variaveis
    ganho_altura_mao = altura_mao[end]-altura_mao[start]
    ganho_altura_CoM = altura_CoM[end]-altura_CoM[start]
    vel_min_CoM = np.min(vel_CoM[:start])
    vel_max_CoM = np.max(vel_CoM[start:end])
    # cria o dataftrame
    disc_df = pd.DataFrame({
        'ganho_altura_maos':ganho_altura_mao,
        'ganho_altura_centro_de_massa':ganho_altura_CoM,
        'velocidade_minima_centro_de_massa_fase_excentrica':vel_min_CoM,
        'velocidade_de_saida_centro_de_massa':vel_max_CoM,
        'fps':fps}, index=[0])
    # salva o dataframe
    with pd.ExcelWriter(path,mode='w') as writer:  
        disc_df.to_excel(writer, sheet_name='discretizado',index=False)
    # salva o dataframe
    with pd.ExcelWriter(path,mode='a') as writer:  
        sinal_df.to_excel(writer, sheet_name='sinal',index=False)
    # adiciona uma aba com as coordenadas
    columns_names = [
        'nose','right eye inner','right eye','right eye outer','left eye inner','left eye','left eye outer',
        'right ear', 'left ear', 'mouth right', 'mouth left', 'right shoulder', 'left shoulder', 'right elbow', 'left elbow',
        'right wrist', 'left wrist', 'right pinky', 'left pinky', 'right index', 'left index', 'right tumb', 'left tumb',
        'right hip', 'left hip','right knee', 'left knee', 'right ankle', 'left ankle', 'right heel', 'left heel','rigth foot index','left foot index'
    ]
    coords_df = pd.DataFrame(coords.reshape(coords.shape[0],coords.shape[1]*2), columns = [m + '_' + a for m in columns_names for a in ['X','Y']])
    # salva o dataframe
    with pd.ExcelWriter(path,mode='a') as writer:  
        coords_df.to_excel(writer, sheet_name='coordenadas',index=False)

def print_limb(frame,limb_markers,color,line=True):
    # para cada membro, plota os marcadores c linhas
    alpha = .3
    # passa pela sequencia consecutiva de pontos plotando as linhas
    if line:
        for i in range(len(limb_markers)-1):
            # Perform weighted addition of the input image and the overlay
            frame = cv2.addWeighted(frame, alpha, cv2.line(frame.copy(),limb_markers[i],limb_markers[i+1],(255,255,255,.1),2, lineType=cv2.LINE_AA), 1 - alpha, 0)
    # plota os pontos individuais
    for marker in limb_markers:
        alpha=.5
        frame = cv2.addWeighted(frame, alpha, cv2.circle(frame.copy(), marker, 3, color, -1, lineType = cv2.LINE_AA), 1 - alpha, 0)

    return frame
# plotagem
def show_coords(coords,CoM,roi,fps=120): 
    fig,ax = plt.subplots(figsize=(roi[2]/50,roi[3]/50))
    # ajusta estetica
    ax.set_ylim(coords[:,:,1].min(),coords[:,:,1].max())
    y_range = coords[:,:,1].max() - coords[:,:,1].min()
    x_range = coords[:,:,0].max() - coords[:,:,0].min() 
    
#     ax.set_xlim(coords[:,:,0].min(),coords[:,:,0].max())
    ax.set_xlim(
        (x_range/2)-(y_range*roi[3]/roi[2])/4,
        (x_range/2)+(y_range*roi[3]/roi[2])/4
    )
    # cria elementos de base
    r_arm, = ax.plot(*coords[0,segmentos['braco_D']+segmentos['antebraco_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_arm, = ax.plot(*coords[0,segmentos['braco_E']+segmentos['antebraco_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    r_leg, = ax.plot(*coords[0,segmentos['coxa_D']+segmentos['perna_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_leg, = ax.plot(*coords[0,segmentos['coxa_E']+segmentos['perna_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    torso, = ax.plot(*coords[0,segmentos['tronco'],:].T,'.-',color='grey',alpha = .5,animated=True)
    # adiciona apresentcao do CoM
    CoM_point, = ax.plot(*coords[0].T,'o',color='orange',alpha = .5,animated=True)
    # Animate function
    def animate(i):
        r_arm.set_data(*coords[i,segmentos['braco_D']+segmentos['antebraco_D'],:].T)
        l_arm.set_data(*coords[i,segmentos['braco_E']+segmentos['antebraco_E'],:].T)
        r_leg.set_data(*coords[i,segmentos['coxa_D']+segmentos['perna_D'],:].T)
        l_leg.set_data(*coords[i,segmentos['coxa_E']+segmentos['perna_E'],:].T)
        torso.set_data(*coords[i,segmentos['tronco'],:].T)
        CoM_point.set_data(CoM[i])
        return [r_arm,l_arm,r_leg,l_leg,torso,CoM_point]
    # animate
    anim=animation.FuncAnimation(fig,animate,frames=len(coords),repeat=True,interval=1000/fps,blit=True)
    return fig,anim

def show_angular_kinematics(coords,roi,fps=120): 
    joint_angles = get_joint_angles(articulacoes,coords)
    # calcula centro de massa
    CoM = np.array([get_CoM(coord) for coord in coords])
    # cria figura base
    fig,ax = plt.subplot_mosaic(
        [['a','aux1'],
         ['a','aux2'],
         ['a','aux3'],
         ['a','aux4']],
        dpi=100, figsize = (8,5)
    )
    ### adiciona a area estatica para plot
    ax['a'].set_ylim(coords[:,:,1].min(),coords[:,:,1].max())
    y_range = coords[:,:,1].max() - coords[:,:,1].min()
    x_range = coords[:,:,0].max() - coords[:,:,0].min() 
    ax['a'].set_xlim(
        -(y_range*roi[3]/roi[2])/4,
        +(y_range*roi[3]/roi[2])/4
    )
    # cria elementos de base
    r_arm, = ax['a'].plot(*coords[0,segmentos['braco_D']+segmentos['antebraco_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_arm, = ax['a'].plot(*coords[0,segmentos['braco_E']+segmentos['antebraco_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    r_leg, = ax['a'].plot(*coords[0,segmentos['coxa_D']+segmentos['perna_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_leg, = ax['a'].plot(*coords[0,segmentos['coxa_E']+segmentos['perna_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    torso, = ax['a'].plot(*coords[0,segmentos['tronco'],:].T,'.-',color='grey',alpha = .5,animated=True)
    # adiciona apresentcao do CoM
    CoM_point, = ax['a'].plot(*coords[0].T,'o',color='orange',alpha = .5,animated=True)
    ### adiciona os dados de angulo articular
    ref_lines = []
    for a,art in zip(['aux1','aux2','aux3','aux4'],['ombro', 'cotovelo','quadril', 'joelho']):
        ax[a].set_title(art.capitalize())
        ax[a].plot(np.linspace(0,len(coords)/fps,len(coords)),joint_angles[art+'_D'],color='b',label='Direita')
        ax[a].plot(np.linspace(0,len(coords)/fps,len(coords)),joint_angles[art+'_E'],color='r',label='Esquerda')
        ax[a].set_ylim(0,210)
        ax[a].set_yticks(range(0,211,45))
        ax[a].spines[['top','right']].set_visible(False)
        ref_line = ax[a].axvline(0,color='k',alpha=.5)
        ref_lines.append(ref_line)
        ax[a].set_xlim(0,None)
        ax[a].set_xticks(np.arange(0,len(coords)/fps,0.5),np.arange(0,len(coords)/fps,0.5) if a == 'aux4' else [])
    ax['aux1'].legend(fontsize='small')
    ax['aux4'].set_xlabel('Tempo (s)')
    # ajuste figura
    plt.tight_layout()
    # Animate function
    def animate(i):
        r_arm.set_data(*coords[i,segmentos['braco_D']+segmentos['antebraco_D'],:].T)
        l_arm.set_data(*coords[i,segmentos['braco_E']+segmentos['antebraco_E'],:].T)
        r_leg.set_data(*coords[i,segmentos['coxa_D']+segmentos['perna_D'],:].T)
        l_leg.set_data(*coords[i,segmentos['coxa_E']+segmentos['perna_E'],:].T)
        torso.set_data(*coords[i,segmentos['tronco'],:].T)
        CoM_point.set_data(CoM[i])
        for ref_line in ref_lines:
            ref_line.set_xdata([i/fps,i/fps])
        return [r_arm,l_arm,r_leg,l_leg,torso,CoM_point,*ref_lines]
    # animate
    anim=animation.FuncAnimation(fig,animate,frames=len(coords),repeat=True,interval=1000/fps*5,blit=True)
    return fig,anim

def show_linear_kinematics(coords,roi,fps=120): 
    # calcula centro de massa
    CoM = np.array([get_CoM(coord) for coord in coords])
    # obtem os dados necessarios
    altura_mao = (coords[:,(15,16),1]).mean(1)
    altura_mao = altura_mao - altura_mao[0]
    altura_CoM = CoM[:,1]
    vel_CoM = np.diff(altura_CoM)*fps
    vel_CoM = np.insert(vel_CoM,-1,0)
    # encontra fase de voo
    start, end = find_fly_phase(coords,fps=120)
    # cria figura base
    fig,ax = plt.subplot_mosaic(
        [['a','aux1'],
         ['a','aux2'],
         ['a','aux3']],
        dpi=100, figsize = (8,5)
    )
    ### adiciona a area estatica para plot
    ax['a'].set_ylim(coords[:,:,1].min(),coords[:,:,1].max())
    y_range = coords[:,:,1].max() - coords[:,:,1].min()
    x_range = coords[:,:,0].max() - coords[:,:,0].min() 
    ax['a'].set_xlim(
        -(y_range*roi[3]/roi[2])/4,
        +(y_range*roi[3]/roi[2])/4
    )
    # cria elementos de base
    r_arm, = ax['a'].plot(*coords[0,segmentos['braco_D']+segmentos['antebraco_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_arm, = ax['a'].plot(*coords[0,segmentos['braco_E']+segmentos['antebraco_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    r_leg, = ax['a'].plot(*coords[0,segmentos['coxa_D']+segmentos['perna_D'],:].T,'.-',color='b',alpha = .5,animated=True)
    l_leg, = ax['a'].plot(*coords[0,segmentos['coxa_E']+segmentos['perna_E'],:].T,'.-',color='r',alpha = .5,animated=True)
    torso, = ax['a'].plot(*coords[0,segmentos['tronco'],:].T,'.-',color='grey',alpha = .5,animated=True)
    # adiciona apresentcao do CoM
    CoM_point, = ax['a'].plot(*coords[0].T,'o',color='orange',alpha = .5,animated=True)
    ### adiciona os dados de cinematica linear
    # altura da mao
    ax['aux1'].set_title("Altura média das maos (m)")
    ax['aux1'].plot(np.linspace(0,len(coords)/fps,len(coords)),altura_mao,color='k')
    ax['aux1'].text(0.1,.8,f'ganho de altura das mãos:\n{altura_mao[end]-altura_mao[start]:.3} m',fontsize='small',
                    transform=ax['aux1'].transAxes)
    # altura do CoM
    ax['aux2'].set_title("Altura do Centro de Massa (m)")
    ax['aux2'].plot(np.linspace(0,len(coords)/fps,len(coords)),altura_CoM,color='k')
    ax['aux2'].text(0.1,.8,f'ganho de altura do centro de massa:\n{altura_CoM[end]-altura_CoM[start]:.3} m',fontsize='small',
                transform=ax['aux2'].transAxes)
    # velocidade do CoM
    ax['aux3'].set_title("velocidade vertical do Centro de Massa (m/s)",fontsize='small')
    ax['aux3'].plot(np.linspace(0,len(coords)/fps,len(coords)),vel_CoM,color='k')
    ax['aux3'].axhline(0,color='grey',alpha=.5)
    ax['aux3'].text(np.argmin(vel_CoM[:start])/fps,np.min(vel_CoM[:start])+.5,f'{np.min(vel_CoM[:start]):.3} m/s',color='g')
    ax['aux3'].plot(np.argmin(vel_CoM)/fps,np.min(vel_CoM),'g.')
    ax['aux3'].text(start/fps,vel_CoM[start]-.5,f'{vel_CoM[start]:.3} m/s',color='g')
    # ajustes gerais
    ref_lines = []
    for a in ['aux1','aux2','aux3']:
        ax[a].axvline(start/fps,color='g',ls='--')
        ax[a].axvline(end/fps,color='r',ls='--')
        ax[a].spines[['top','right']].set_visible(False)
        ax[a].set_xlim(0,None)
        ref_line = ax[a].axvline(0,color='k',alpha=.5)
        ref_lines.append(ref_line)
    # ajuste figura
    plt.tight_layout()
    
    # Animate function
    def animate(i):
        r_arm.set_data(*coords[i,segmentos['braco_D']+segmentos['antebraco_D'],:].T)
        l_arm.set_data(*coords[i,segmentos['braco_E']+segmentos['antebraco_E'],:].T)
        r_leg.set_data(*coords[i,segmentos['coxa_D']+segmentos['perna_D'],:].T)
        l_leg.set_data(*coords[i,segmentos['coxa_E']+segmentos['perna_E'],:].T)
        torso.set_data(*coords[i,segmentos['tronco'],:].T)
        CoM_point.set_data(CoM[i])
        for ref_line in ref_lines:
            ref_line.set_xdata([i/fps,i/fps])
        return [r_arm,l_arm,r_leg,l_leg,torso,CoM_point,*ref_lines]
    # animate
    anim=animation.FuncAnimation(fig,animate,frames=len(coords),repeat=True,interval=1000/fps*5,blit=True)
    
    return fig,anim

