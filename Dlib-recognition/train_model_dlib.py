import dlib
import os
import numpy as np
import cv2


# 获取所有文件（人脸id）
def get_face_list(path):
    for root,dirs,files in os.walk(path):
        if root == path:
            return dirs



if __name__ == "__main__":
    # 加载人脸特征提取器
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 加载人脸标志点检测器
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    # 人脸识别训练图片存放路径
    base_path = "../face_collect/"
    
    # 获取人脸 id 列表
    face_list = get_face_list(base_path)
    
    # 用来存储人脸特征和人脸id的列表
    list_face_vector = []
    list_face_id =[]
    
    
    for face_id in face_list:
        
        for f_img in os.listdir(os.path.join(base_path,face_id)):
            if f_img.endswith(".jpg"):
                file_img = os.path.join(base_path,face_id,f_img)
                
                print("Extract face vector for file %s"%(file_img))
                # 读取图像并转换为RGB
                img = cv2.imread(file_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 在整图内检测标志点
                img = np.array(img)
                h,w,_ = np.shape(img)
                rect = dlib.rectangle(0,0,w,h)
                shape = sp(img, rect)
                
                # 获取128维人脸特征
                face_vector = facerec.compute_face_descriptor(img,shape)
                
                # 特征 和 id 保存
                list_face_vector.append(face_vector)
                list_face_id.append(face_id)
    
    # 将最终结果进行保存
    face_vectors = np.array(list_face_vector)
    ids = np.array(list_face_id)

    # 模型保存
    np.savez('trainer',face_vectors=face_vectors,ids=ids)
    
                



                
      