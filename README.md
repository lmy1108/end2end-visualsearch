# end2end-visualsearch
A visual search system used for product retrieval, test on amazon product data

## Purpose of design
This Project designs an online image retrieval system. Image preprocessing is performed by target detection and image segmentation, which solves the problem of acquiring key information in complex background. The integrated ResNet and HNSW methods are proposed to extract image features and perform nearest neighbor search. The results of target detection and image tags are The combination of inverted indexes enables further refined merchandise search. At the same time, this paper also proposes a retrieval system based on image classifier based on convolutional neural network, which further optimizes the accuracy and speed of retrieval through commodity pre-classification.<br>
The system is based on Python+Mxnet+Flask. The front end interacts with the background through the web interface, providing a variety of image search methods that meet the needs of modern e-commerce websites. The background includes offline and online processing. Offline processing completes feature extraction, classification, and tag indexing of all products. Online processing completes functions from image preprocessing to feature extraction and nearest neighbor search. For the real-time nature of the system, multiple concurrent high-availability processing is performed in the background using the Redis cache. The system uses the Amazon e-commerce data set for testing, which enables a more accurate search than the general e-commerce search system under the condition of real-time performance.

## demo


### FrontEnd Visualization
Front End uses javascript to make mesh results visualize according to the given point cloud and parameters given.

### Feature Extraction and KNN search

According to ResNet and extracting image features: download the image and name the image as its ASIN ID through the address in the source file provided by the Amazon dataset, configure the CUDA environment on the Linux system and perform GPU-based Mxnet installation, loading Pre-train the ResNet model, convert each picture into the required input format. After the information is completed, perform batch feature extraction, and put 1 million pictures into the pre-trained ResNet one by one to establish an ASIN-to-feature mapping.
<br>
The nearest neighbor search of this system mainly uses an improved NSW (Navigable Small World) algorithm.

### Object dectection module

For some products with more complicated patterns such as clothing, bags, shoes, etc., you can consider using a more sophisticated image search method, that is, adding some pre-processing before feature extraction. As mentioned in the above, the system integrates target detection. And image segmentation two image preprocessing algorithms to improve accuracy.
<br>In terms of target detection, the system uses the pre-trained Faster R-CNN structure of the COCO training set to target the target and the classification label of the target, while in the image segmentation, the FCN is used to segment the most critical elements in the image.

<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs9.png" alt="Sample">
	<p align="center">
		<em>Use Faster RCNN to detect person</em>
	</p>
</p>

### Object Detection & Inverted Index Based Searching

In many cases, there are two products with similar patterns and similar shapes, but they are not the same type of goods in nature. In this case, only the feature extraction may be misjudged, so the system also uses another search method. - Image search based on target detection and inverted index.
<br>

In actual use, the system automatically recognizes the images uploaded by the user and performs preliminary target recognition, searching and matching in the documents created by the image set.
<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs12.png" alt="Sample">
	<p align="center">
		<em>use words to generate more accurate search</em>
	</p>
</p>
<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs13.png" alt="Sample">
	<p align="center">
		<em>women watch only with same looks</em>
	</p>
</p>

<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs14.png" alt="Sample">
	<p align="center">
		<em>silver shoes with same pattern</em>
	</p>
</p>
### Product Classification Module
In the retrieval process of goods, it is often found that the results of using image retrieval do not belong to the user's product category. Although these products are very similar in appearance to the products that users search for, they are completely different types of goods. For example, when searching for a certain style of clothing, they often search for books with very similar pattern covers. From the customer's point of view, Such a search is obviously not ideal. The wrong category will make the goods that the customer wants are very different from the returned goods, so that the customer's confidence in searching in this way is reduced, and finally the utility of the system is reduced, in order to change the situation. This system uses two methods to solve the classification problem.
<br> We use ResNet to extract features from 500k datas and trained an ensenbled model to do the classification.
<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs2.PNG" alt="Sample">
	<p align="center">
		<em>train the random forest model</em>
	</p>
</p>
<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs4.png" alt="Sample">
	<p align="center">
		<em>the result shows the clothing catagory wins</em>
	</p>
</p>

<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs5.png" alt="Sample">
	<p align="center">
		<em>the result shows the mobile catagory wins</em>
	</p>
</p>
### Cache Supporting High Concurrency  
This system uses Redis as a cache to solve the concurrency problem. The main idea is to create a List in the cache, and use the idea of analog queue to gradually increase the picture of different users to the end of the List, using multi-thread asynchronously from the processing. The List header takes data so that users who are largely guaranteed will not interfere with each other when they access it at the same time. We also conducted stress tests to estimate the maximum number of visits. Use the Curl command to iterate through the target URL and record the number of visits per second. In the case of the current system memory size, the system can support about 200 simultaneous requests.
<p align="center">
	<img src="https://github.com/lmy1108/end2end-visualsearch/blob/master/images/vs15.PNG" alt="Sample">
	<p align="center">
		<em>cache structure</em>
	</p>
</p>
