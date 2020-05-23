<h1>Object Detection using SSD</h1>
<p>In this project we are going to implement a system which use CNN to detect objects in a picture using SSD (Single-Shot MultiBox Detector) algorithm</p>
<p>SSD is good in both speed of detection and accuracy</p>
<h1>Goal</h1>
<p>Desing and implement a system using SSD algorithm to detect objects in a picture or video real time.</p>
<h1>Project Outline</h1>
<ul>
  <li>Object localization (we will see how to combine classification and regression).</li>
  <li>Broden our scope from object localiztion to object detection</li>
  <li>Sliding windows efficient implementation.</li>
  <li>Problem of scale, we will deal with this problem with caused by the distance of objects in a picture from the point which has taken</li>
  <li>SSD architecture for industrial usage</li>
  <li>Modify the SSD algorithm to work on videos.</li>
  <li>Jaccard index(IoU), non-max suppression</li>
  </ul>
  <h3>Object Localization</h3>
  <p>In this concept, we don't just want to know what are in the image, we want to know where they are</p>
  <p>To aim this purpose, we need five logistic regression on top of the ResNet for detecting class, x center, y center, height ,and width.</p>
  <h5>Loss Function</h5>
  <p>Our loss function includes three parts</p>
  <ol>
  <li>Binary Cross Entropy: p(object | image): this part tells us whether or not there is even an object in the image.</li>
  <li>Categorical Cross Entropy: p(class 1 | image), p(class 2 | image) ... p(class k | image): this part tells us which class objects belong to</li>
  <li>MSE : in this part we have four regression output for bounding box(CX, CY, Height, Width)(should not contribute to loss when there is no object in the image)</li>
  </ol>
  <h3>Object Detection</h3>
  <p>This is a generalized version of object localization. In this concept we may have 0 or several objects within an image. The goal is to detect all of them and draw rect around each object.</p>
  <ul>
  <li>Worth thinking about: what kind of data structures do we need?</li>
  <li>A CNN must output a fixed set of numbers</li>
  <li>But an image may have 0 objects, or it may have 50- how can it output the right numbers for all cases?</li>
  <li>Naive strategy: in a loop</li>
  <ul>
    <li>Look for object with highest class confidence</li>
    <li>Output its p(class | image), cx, cy, height, width</li>
    <li>Erase that object from the image</li>
  </ul>
  </ul>
  <h5>How can Find Objects in an Image?</h5>
  <p>Sliding window technique: take some window and for each position in the original image pass this sub-image to the CNN. One of the major problem of this method is its low speed, O(N^2). To solve this problem we would use convolution operation.</p>
  <p>SSD: The main concept is that by using CNN we would get same result as sliding window by passing the image through CNN just one time, that's why its name is single-shot. One more advantagous of this algorithm is that there is no need to tell the CNN which regions may have objects</p>
<h3>The Problem of Scale</h3>
<p>There are objects that may seem very small because of their distance to the camera, how can solve this problem?</p>
<p>The general pattern of CNN is that you go through each layer the image is shirinking and therefore the features you are finding go from small to big. The idea is attach mini-neural network to intermediate layers of a pre-trained network. For each output we will do object detection separately.</p>
<h3>The Problem of Shape (Aspect Ratio)</h3>
<ul>
<li>Windwo Size: In a picture there are objects with different sizes, for example people are tall and cars are wide, so what size should the window be?</li>
<li>We might be looking at a window where both objects might appear in the same window with one occluding the other.</li>
  <li>Different angle of an object: for example a person may lay down</li>
</ul>
<p>Solution is: instead of one window, use default boxes in each position, for each rect we try to detect an object by passing it through our CNN</p>
<p>We not only look at the image at multiple scales but we apply each box to each window at each scale</p>

<h3>Start Running the Project</h3>
<ol>
<li>Download the tensorflow/models repository: <code>git clone https://github.com/tensorflow/models.git</code></li>
  <li>Start Notebook inside research/object_detection folder</li>
  <li>Install Protocol Buffers: (windows) <code>conda install -c anaconda protobuf</code> To ensure about correct installation <code>protoc --version</code></li>
  <li>Run this from the "research folder": <code>protoc object_detection/protos/*.proto --python_out=.</code>
  <li>Exmaple command for an image: <code>python main.py --content image --path "./sea.jpg"</code></li>
    <li>Exmaple command for a video: <code>python main.py --content video --path "./traffic.mp4"</code></li>
</ol>
<h4>Sample Output for Detecting Objects in an Image</h4>
<img src="https://github.com/amoazeni75/object-detection-ssd/blob/master/sea_output.png" alt="sea"/>
<h4>Sample Output for Detecting Objects in a Video</h4>
<video controls>
  <source src="movie.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
