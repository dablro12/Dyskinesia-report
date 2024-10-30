import os
def avi2mp4(results, model_name, exp_save_dir = '/home/eiden/eiden/pd-ai/experiments/test1'):
    os.makedirs(exp_save_dir, exist_ok= True)
    
    metadata = {
        "filename" : results[0].path.split('/')[-1].split('.mp')[0],
        "avi_save_dir" : results[0].save_dir,
        "our_save_dir" : exp_save_dir,
        "speed" : results[0].speed,
        "orig_shape" : results[0].orig_shape,
    }
    
    save_avi = os.path.join(metadata['avi_save_dir'], metadata['filename']+'.avi')
    save_mp4 = os.path.join(metadata['our_save_dir'], metadata['filename']+'-'+model_name+'.mp4')
    
    # avi 데이터를 mp4로 변환 (ffmpeg 사용) * background에서 실행
    print(f"avi2mp4: {save_avi} -> {save_mp4}")
    os.system(f"ffmpeg -i {save_avi} {save_mp4}")
    os.remove(save_avi)