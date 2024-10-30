from ultralytics import YOLO
import os

class yolo_PoseEstimation:
    def __init__(self, model_name: str):
        """
        Initialize the YOLO model for pose estimation.

        :param model_name: Path to the YOLO model file (e.g., "yolo11n-pose.pt")
        """
        self.model = self.YOLO_model_loader(model_name)
        
    def YOLO_model_loader(self, model_name: str):
        """
        Load the YOLO model.

        :param model_name: Path to the YOLO model file
        :return: Loaded YOLO model
        """
        YOLO_Estim_model = YOLO(model_name)
        return YOLO_Estim_model
    
    def predict(self, video_path: str, save: bool = False, save_dir: str = "output"):
        """
        Perform pose estimation on the input video and optionally save the annotated video.

        :param video_path: Path to the input video file
        :param save: Boolean indicating whether to save the annotated video
        :param save_dir: Directory where the annotated video will be saved
        :return: Results object containing prediction details
        """
        # Ensure the save directory exists
        if save:
            os.makedirs(save_dir, exist_ok=True)
        
        # Perform prediction
        results = self.model(
            source=video_path,
            stream=False,
            save=save,            # Save annotated output
            save_txt=False,       # Do not save detection results as text
            save_conf=False,      # Do not save confidence scores
            project=save_dir,     # Directory to save results
        )
        
        return results
    
if __name__ == "__main__":
    yolo_class = yolo_PoseEstimation(model_name = "yolo11n-pose.pt")

    # Define the input video path and the directory to save the output
    input_video = "data/video/sample_2_1_3_clip.mp4"
    output_directory = "data/result"

    # Perform prediction and save the annotated video
    results = yolo_class.predict(
        video_path=input_video,
        save=True,
        save_dir=output_directory
    )