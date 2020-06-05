# yuchen-PhotoWakeUp
An implementation of CVPR2019 paper - PhotoWake-Up: 3D Character Animation from a Single Photo <br>
<br>
The original project website (and paper): https://grail.cs.washington.edu/projects/wakeup/ <br>
Many thanks to the original authors contributions! <br>
<br>
The implementation of section 4.1. Mesh Warping, Rigging, & Skinning of the original paper is complete. With an input including human in front view and without self-occlusion, a 3D-model that matches the boundary of the input human area could be constructed. Texture, animation and background could be added using Maya2019. <br>
<br>
The implementation of section 4.2 Self-occlusion is not completed. File smpl_seg includes the per-vertex label of smpl model, and codes in file labelmap could be used to generate and warp body label map. However, those final steps to deal with self-occlusion are not implemented in this project. <br>
# Examples
![](https://github.com/yczhang-seu/yuchen-PhotoWakeUp/raw/master/result/WakeUp1_front.gif width="200")
![](https://github.com/yczhang-seu/yuchen-PhotoWakeUp/raw/master/result/WakeUp1_side.gif)
<br>
# How to use
1. Find a proper input image. The position of the human needs to nearly be at the center, and the human should not include self-occlusion. Use the modified SPIN to generate the cropped input (larger input will be cropped to 224* 224), the depth maps and the smpl models. 
2. Generate masks of the cropped input image and smpl models, this could be manually done by Adobe Photoshop.<br>
3. Put the masks and depth maps in file data. Modify the dirs in main.py, then run the code. To check if the contours are matched properly, use pc.dispCorres(), and if the matching is not good enough, increase the last param of pc.dpBoundarymatch() （e.g. from 32 to 64).<br>
4. Check the model in file output. The obj file could be displayed and modified in Maya2019. To change the thickness of the model, change the related param in outputvert(). If you want to generate a new obj model, please delete the old one first. <br>
5. Use Maya2019 to add texture and background, make animation, etc.<br> 

# Please note
As a beginner project in python, the result is far from perfect. The generated model is quite coarse, and further manual adjustments are needed in most cases. I am glad to know if you have any suggestions to make the project better!
