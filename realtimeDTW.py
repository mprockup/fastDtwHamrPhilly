import sys
import getopt
import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt
import librosa
import pdb


# seach ahead and behind current frame estimate for the next closest frame 
searchAhead = 5
searchBehind = 1
searchWidth = searchBehind + 1 + searchAhead
testSequenceLength = 200

# fake [searchWidth x 12] array of sequential inputFrames
exampleMultipleFramesInput = np.random.rand(searchWidth,12)


# fake 12 dim pitch class vector (chroma) at current live audio
# (last) element of input frames over time
exampleFrameCurrentInput = exampleMultipleFramesInput[0,-1] 

# fake [searchWidth x12] array of seachable pitch class vectors in reference audio
exampleMultipleFramesReference = np.random.rand(searchWidth,12)


def getDifference(inputFeatureVector,referenceSearchMatrix,differenceMetric = 'sum'):
	# DOC STR
	''' 
		getDifference(inputFeatureVector,referenceSearchMatrix,differenceMetric = 'sum'):

			gets differece scalar of two feature vectors or vector of scalar differeces 
			over multipleFeatureVectors

		IN: 
			inputFeatureVector 		:	featureVector at single time step
			referenceSearchMatrix	:  	matrix possible corresponding frames in referece sequence
			differenceMetric		:	calculate distance between frames over all feature dimentions.
										'sum' differences over features.
										'product' differences over features
										'mean' 	   differences over features
										'median'   differences over fearures
										default = 'sum'
	 
		OUT: 
			difference				:	return a vector of differences from the inputFeatureVector to all 
										frames in the referenceSearchMatrix.
	'''

	# get rows of seach matrix
	referenceSearchSize = referenceSearchMatrix.shape[0]

	# tile inputVector to match size of seachMatrix
	inputVectorTiled = np.tile(inputFeatureVector,(referenceSearchSize,1))

	#get absolute difference between frames
	differenceMatrix = np.abs(inputVectorTiled - referenceSearchMatrix);

	# pdb.set_trace();

	# compute combined difference over all seach frames
	if(differenceMetric == 'sum'):
		differenceOverFrames = np.sum(differenceMatrix, axis=1)
	elif(differenceMetric == 'product'):
		differenceOverFrames = np.product(differenceMatrix, axis=1)
	elif(differenceMetric == 'mean'):
		differenceOverFrames = np.mean(differenceMatrix, axis=1)
	elif(differenceMetric == 'median'):
		differenceOverFrames = np.median(differenceMatrix, axis=1)
	else: #default = 'sum'
		differenceOverFrames = np.sum(differenceMatrix, axis=1)

	return differenceOverFrames


def getCost(differenceVectorWindows, currentEstimates, estimateOffset=searchBehind, penalty='sum'):
	# DOC STR
	''' 
		getCost(differenceVectorWindows,currentEstimates, penalty='sum'):
			evaluates the least cost path through the distance matrix formed by the 
		IN: 
			differenceVectorWindows	:	widowed frames of distance matrix
			currentEstimates		:  	currently estimated alignment locations
			penalty					:	accumulate a penalty for moving outside adjacent frame
										'sum' additive penalty to move to newFrame outside 'currentFramePos + 1'
										'product' multiplicative penalty to move to outside 'currentFramePos + 1'
										default = 'sum'
		OUT: 
			costPath				:	return the path through the reference 
			cost					:	return the calculated cost
	'''
	differenceVectorWindows = np.array(differenceVectorWindows)
	currentEstimates = np.array(currentEstimates)
	frameBoundry = currentEstimates - estimateOffset

	cost = np.zeros(differenceVectorWindows.shape)
	updatedEstimates = np.zeros(currentEstimates.shape)
	updatedEstimates[-1] = currentEstimates[-1]

	for i in range(differenceVectorWindows.shape[0]-1,0,-1):
		tempFrame = differenceVectorWindows[i-1,:]
		relativePositionInPrevious = updatedEstimates[i] + (frameBoundry[i]-frameBoundry[i-1])
		
		# pdb.set_trace();
		cost[i,:relativePositionInPrevious] = np.flipud(np.cumsum(np.flipud(tempFrame[:relativePositionInPrevious])))
		cost[i,(relativePositionInPrevious-1):] = np.cumsum(tempFrame[(relativePositionInPrevious-1):])
		updatedEstimates[i-1] = np.argmin(cost[i,:]) + frameBoundry[i-1]
		

	return (updatedEstimates, cost)



def main():
	# parse command line options
	try:
		opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
	except getopt.error, msg:
		print msg
		print "for help use --help"
		sys.exit(2)
	# process options
	for o, a in opts:
		if o in ("-h", "--help"):
			print __doc__
			sys.exit(0)

	calcFeatures = True
	if(calcFeatures):
		file1 = 'AtlantaHigdonConcertoForOrchestraMov2.wav'
		file2 = 'PhillyHigdonConcertoForOrchestraMov2.wav'
		
		print 'loading: ' + file1
		y1, sr1 = librosa.load(file1,sr=44100)
		print '...done'
		
		print 'loading: ' + file2
		y2, sr2 = librosa.load(file2,sr=44100)
		print '...done'

		nfft = 8192
		C1 = np.transpose(librosa.feature.chromagram(y=y1, sr=sr1,n_fft=nfft,hop_length=nfft))
		C2 = np.transpose(librosa.feature.chromagram(y=y2, sr=sr2,n_fft=nfft,hop_length=nfft))
		
		fig1 = 0;
		fig1 = plt.figure(1,figsize=(8 , 4));
		plt.imshow(np.transpose(C1),aspect='auto',interpolation='nearest')
		plt.savefig('C1Features.png')
		fig1.clf()

		fig1 = 0;
		fig1 = plt.figure(1,figsize=(8 , 4));
		plt.imshow(np.transpose(C1[:100,:]),aspect='auto',interpolation='nearest')
		plt.savefig('C1FeaturesSmall.png')
		fig1.clf()

		fig1 = 0;
		fig1 = plt.figure(1,figsize=(8 , 4));
		plt.imshow(np.transpose(C2),aspect='auto',interpolation='nearest')
		plt.savefig('C2Features.png')
		fig1.clf()

		fig1 = 0;
		fig1 = plt.figure(1,figsize=(8 , 4));
		plt.imshow(np.transpose(C2[:100,:]),aspect='auto',interpolation='nearest')
		plt.savefig('C2FeaturesSmall.png')
		fig1.clf()

		np.save('C1',C1)
		np.save('C2',C2)

	C1 = np.load('C1.npy')
	C2 = np.load('C2.npy')

	# pdb.set_trace();

	doRealTimeDTW(testSequenceReference=C2,testSequenceLive=C1)

def doRealTimeDTW(testSequenceReference,testSequenceLive=None,featureMethod):
	#start the DTW audio alignment algorithm

	# the algorithm is widowed, keep track of position of estimates as they relates to entire sequence
	blindFrameAlignmentEstimate = [];

	blindFrameAlignmentEstimate.append(searchBehind)

	#initialize the location of the previous position
	#in this case its the weighed center of 'searchBehind + searchAhead'
	previousPosition = searchBehind

	blindFrameAlignmentEstimate.append(searchBehind)

	# windowedFrameList = [];
	differenceVectorList = []
	# ===================================
	# 	START Listen and Align Loop
	# ===================================
	# loop through temp features
	for inFrameCnt in range(searchBehind + 1, testSequenceLive.shape[0] - (searchAhead + 1)):
		
		#get a feature frame of audio input
		# liveAudioFeatures = exampleFrameCurrentInput;
		liveAudioFeatures = testSequenceLive[inFrameCnt,:]
		
		if(previousPosition<searchBehind):
			previousPosition = searchBehind
		
		# get a subset of the available reference frames for alignment based on search frame
		searchFrame = testSequenceReference[(previousPosition-searchBehind):(previousPosition+searchAhead),:]

		# get differece of current frame features and frames of reference features  
		differenceVector = getDifference(liveAudioFeatures,searchFrame,differenceMetric = 'sum')
		print str(differenceVector.shape[0])

		# pdb.set_trace()
		# get the relative subset frame position of the minimum difference 
		relativeEstimatedPos = np.argmin(differenceVector)
	
		# transform relative frame postion to absolute position
		# remove artificial offset added that allowed us serching the past
		absoluteEstimatedPosition = relativeEstimatedPos + previousPosition - searchBehind
		
		print 'frame: ' + str(inFrameCnt) + '\tPrevPos: ' + str(previousPosition) + '\tRelative Position: ' + str(relativeEstimatedPos) + '\tAbsolute Position: ' + str(absoluteEstimatedPosition)

		# print '...Relative Position: ' + str(relativeEstimatedPos) + '...Absolute Position: ' + str(absoluteEstimatedPosition)

		# pdb.set_trace()

		# if there is multiple minimums, from argmin, there are multiple estimates
		# use the one closest to the previous position
		# this is a special case, it should happen rarely
		# closestPos = 0
		# estimateTemp = absoluteEstimatedPosition
		# if(absoluteEstimatedPosition.size > 1):
		# 	posDiff = 1000000 #inf  !!! find a better way to do this.
		# 	for i in range(0,len(estimateTemp)):
		# 		if(np.abs(previousPosition-estimateTemp[i])<posDiff):	#find smallest index distance from previous frame
		# 			posDiff = np.abs(previousPosition-estimateTemp[i]) 	#set new min dist from previous frame
		# 			closestPos = i
		# 			absoluteEstimatedPosition = estimateTemp[closestPos]												#save index
		
		blindFrameAlignmentEstimate.append(absoluteEstimatedPosition)
	
		previousPosition = absoluteEstimatedPosition
	
		# add current frame to 
		differenceVectorList.append(differenceVector)
	
		# estimates, costVector = getCost(differenceVectorList,blindFrameAlignmentEstimate)

		# pdb.set_trace()
	# print blindFrameAlignmentEstimate
	# fig1.clf()
	# fig1 = 0;
	fig1 = plt.figure(1,figsize=(10 , 10));
	plt.plot(blindFrameAlignmentEstimate)
	plt.xlabel('live frames')
	plt.ylabel('aligned reference trames')
	plt.savefig('alignment.png')
	fig1.clf()

# ===================================
# 	END Listen and Align Loop
# ===================================

if __name__ == "__main__":
	main()

