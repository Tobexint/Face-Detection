# Face-Detection
Web app pretrained against a database where users can upload images to detect the individual faces in the images.

## Objs
- Initialize a pre-trained MTCNN model from facenet_pytorch.
- Detect faces in an image using MTCNN model.
- Display the resulting bounding boxes of faces detected by the model.
- Crop out detected faces for further analysis.
- Determine facial landmarks such as eyes, nose, and mouth using the MTCNN model.
