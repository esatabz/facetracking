import cv2
import os
import dlib
import numpy as np

def face_recognize(face_vec,face_dataset,ids):
    N = face_dataset.shape[0]
    diffMat = np.tile(face_vec,(N,1))-face_dataset
    
    # 计算欧式距离
    distances = np.linalg.norm(diffMat,axis=1)
    
    # 找到最小距离
    idx = np.argmin(distances)
    
    # 返回id编号与距离
    return ids[idx],distances[idx] 
    
    
def face_recognize_cos(face_vec,face_dataset,ids):
    N = face_dataset.shape[0]
    a = np.tile(face_vec,(N,1))
    b = face_dataset
    
    # 计算cos 距离 
    dis_cos = np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    
    # 找到最大距离
    idx = np.argmax(dis_cos)
    
    # 返回id编号与距离
    return ids[idx],dis_cos[idx] 
    
    
    
    


if __name__ == "__main__":
    # 加载训练好的人脸模型
    model = np.load('trainer.npz')
    
    face_vectors = model['face_vectors']
    face_ids = model['ids']
    
    print(face_vectors)
    print(face_ids)
    
    # Dlib 人脸检测器
    detector = dlib.get_frontal_face_detector()

    # Dlib 标志点检测器 
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    # Dlib 人脸特征提取器
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:  
        # 读取一帧图像
        success, img = cap.read()
        
        if not success:
            continue
        
        # BGR 转 gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)
        
        # 进行人脸检测
        dets = detector(img_gray, 1)
       
        # 遍历检测的人脸
        for k, d in enumerate(dets):
            
            # 画框
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 3)
            
            # 标志点检测
            shape = sp(img_rgb, d)
            
            # 获取人脸特征
            face_vector = facerec.compute_face_descriptor(img_rgb, shape)
            
            # # 进行识别返回ID与距离
            face_id, dis = face_recognize(np.array(face_vector),face_vectors,face_ids)
            
            if (dis < 0.45):
                str_out = "%s  %.2f"%(face_id,dis)
            else:
                str_out = "unknown %.2f"%(dis)
        
            # # 进行识别返回ID与距离
            # face_id, dis = face_recognize_cos(np.array(face_vector),face_vectors,face_ids)
            
            # if (dis > 0.95):
                # str_out = "%s  %.2f"%(face_id,dis)
            # else:
                # str_out = "unknown %.2f"%(dis)
           
            # 检测结果文字输出
            cv2.putText(img, str_out, (d.left()+5,d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           
        # 显示检测结果
        cv2.imshow("FACE",img)
        
        # 按键 "q" 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
    cap.release() 