import java.util.Random;
import java.util.ArrayList;



public class BackPropagation extends SupervisedLearner
{
	private final int NODES_PER_LAYER = 100;
	private final int C_LAYERS = 2;		// specify the number of hidden layers
	private final boolean USE_MOMENTUM = false;
	private final double MOMENTUM_RATE = 0.1;
	private final double TRAINING_PERCENT = 0.80;
	private final double LEARNING_RATE = 0.20;		// specify the learning rate
	private final int MIN_EPOCHS = Math.max(4 * C_LAYERS * NODES_PER_LAYER, Math.max(NODES_PER_LAYER*3, 30));
	private final int MAX_EPOCHS_WITHOUT_IMPROVEMENT = 5;
	Random random;
	ArrayList<ArrayList<Node>> rgLayers;	
	int bias = 1;
	
	public BackPropagation(Random random){
		this.random = random;
	}
	
	private void makeLayer(int cInputNodes, int cOutputNodes, int cHiddenLayers){
		rgLayers = new ArrayList<ArrayList<Node>>(); 
		
		int[] cNodesPerLayer = new int[cHiddenLayers + 2];
		
		for(int i = 0;  i< cNodesPerLayer.length; i++){
			int cNodes = 0;
			if(i == 0){ //input
				cNodes = cInputNodes;
			}
			else if(i == cHiddenLayers + 1){ //output
				cNodes = cOutputNodes;
			}
			else{ //hidden
				cNodes = NODES_PER_LAYER;
			}
			
			cNodesPerLayer[i] = cNodes;
		}
		
		for(int i = 0; i <= cHiddenLayers; i++){
			ArrayList<Node> iLayer = new ArrayList<Node>();			
			for(int j = 0; j < cNodesPerLayer[i+1]; j++){
				Node tNode = new Node(cNodesPerLayer[i] + 1);
				iLayer.add(tNode);
			}
			rgLayers.add(iLayer);
		}			
	}

	public void propagateForwardPath(Matrix features, int a){ // where a is row of matrix
		for(int i = 0; i < rgLayers.size(); i++){
			if(i == 0){ 		// first hidden layer	
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias * rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.cols(); k++){
						net += features.get(a, k) * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1/(1+Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
			else{
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < rgLayers.get(i).get(j).getNumWeight() - 1; k++){
						net += rgLayers.get(i - 1).get(k).getOutput() * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
		}	
	}
	
	public double propagateBackwardPath(Matrix features, Matrix labels, int a){

		double momentum = USE_MOMENTUM ? MOMENTUM_RATE : 0;
		double mse = 0;
		for(int i = rgLayers.size()-1; i >= 0; i--){
			if(i == rgLayers.size()-1){ 		// output layer
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double output = rgLayers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					int dAnswer = -1;
					if(labels.get(a, 0) == j){
						dAnswer = 1;
					}
					else{
						dAnswer = 0;
					}
					double delta = (dAnswer-output) * fnet;
					
					mse += Math.pow((dAnswer-output), 2)/2.0;
					
					rgLayers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < rgLayers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (LEARNING_RATE * delta * bias) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						else
							change = (LEARNING_RATE * delta * rgLayers.get(i-1).get(k-1).getOutput()) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						rgLayers.get(i).get(j).setWeightChangeElement(k, change);
					}
				}
			}

			else if(i > 0){			// hidden layer
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double output = rgLayers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					double sum = 0;
					for(int k = 0; k < rgLayers.get(i + 1).size(); k++){
						sum += rgLayers.get(i + 1).get(k).getDelta() * rgLayers.get(i + 1).get(k).getWeightElement(j + 1);
					}
					double delta = fnet * sum;
					rgLayers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < rgLayers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (LEARNING_RATE * delta * bias) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						else
							change = (LEARNING_RATE * delta * rgLayers.get(i - 1).get(k - 1).getOutput()) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						rgLayers.get(i).get(j).setWeightChangeElement(k, change);
					}						
				}
			}
			else{			// first layer	
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double output = rgLayers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					double sum = 0;
					for(int k = 0; k < rgLayers.get(i + 1).size(); k++){
						sum += rgLayers.get(i + 1).get(k).getDelta() * rgLayers.get(i + 1).get(k).getWeightElement(j + 1);
					}
					double delta = fnet * sum;
					rgLayers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < rgLayers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (LEARNING_RATE * delta * bias) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						else
							change = (LEARNING_RATE * delta * features.get(a, k - 1)) + (momentum * rgLayers.get(i).get(j).getWeightChangeElement(k));
						rgLayers.get(i).get(j).setWeightChangeElement(k, change);
					}						
				}					
			}
		}
		return mse;
	}

	public void updateWeights(){
		for(int i = 0; i < rgLayers.size(); i++){
			for(int j = 0; j < rgLayers.get(i).size(); j++){
				for(int k = 0; k < rgLayers.get(i).get(j).numWeight; k++){
					double newWeight = rgLayers.get(i).get(j).getWeightElement(k) + rgLayers.get(i).get(j).getWeightChangeElement(k);
					rgLayers.get(i).get(j).setWeightElement(k, newWeight);
				}
			}
		}
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		System.out.println("Training");
		int nInput = features.cols(); 
		int nOutput = labels.valueCount(0);
		makeLayer(nInput, nOutput, C_LAYERS);		// a function to set up the hidden layers
		
		int trainingSize = (int)(TRAINING_PERCENT * features.rows());

		ArrayList<Double> trainMSE = new ArrayList<>();
		ArrayList<Double> testMSE = new ArrayList<>();
		ArrayList<Double> accuracies = new ArrayList<>();
		int cEpochs = 0;
		int cEpochsWithoutImprovement = 0;
		boolean loop = true;
		
		do{	
			features.shuffle(random, labels);
			Matrix mxTrainFeatures = new Matrix(features, 0, 0, trainingSize, features.cols());
			Matrix mxTrainLabels = new Matrix(labels, 0, 0, trainingSize, 1);
			Matrix mxTestFeatures = new Matrix(features, trainingSize, 0, features.rows()-trainingSize, features.cols());
			Matrix mxTestLabels = new Matrix(labels, trainingSize, 0, features.rows()-trainingSize, 1);
			double fpMSE = 0;
			for(int i = 0; i < mxTrainFeatures.rows(); i++)
			{
				propagateForwardPath(mxTrainFeatures, i);
				fpMSE += propagateBackwardPath(mxTrainFeatures, mxTrainLabels, i);
				updateWeights();
			}
			fpMSE = fpMSE/mxTrainFeatures.rows();
			trainMSE.add(fpMSE);
			
			fpMSE = 0;
			for(int i = 0; i < mxTestFeatures.rows(); i++)
			{	
				propagateForwardPath(mxTestFeatures, i);
				fpMSE += propagateBackwardPath(mxTestFeatures, mxTestLabels, i);
				updateWeights();
			}
			fpMSE = fpMSE/mxTestFeatures.rows();
			testMSE.add(fpMSE);
			
			double currentAccuracy = calculateAccuracy(mxTestFeatures, mxTestLabels);
			
			
			if(cEpochs > MIN_EPOCHS)
			{
				double lowest = 1;
				for(int i = 0; i < MIN_EPOCHS; i++)
				{
					if(accuracies.get(accuracies.size() - 1 - i) < lowest){
						lowest = accuracies.get(accuracies.size() - 1 - i);
					}
				}
				if(lowest > currentAccuracy)
					cEpochsWithoutImprovement = 0;
				else
					cEpochsWithoutImprovement++;
				
				if(cEpochsWithoutImprovement > MAX_EPOCHS_WITHOUT_IMPROVEMENT)
					loop = false;

			}
			accuracies.add(currentAccuracy);
			cEpochs++;
			if(cEpochs%10==0){
				System.out.println("Epoch: "+ cEpochs);
				System.out.println("Epochs without improvement: " + cEpochsWithoutImprovement);
			}
		}while(loop);
		
		System.out.println("final train mse: " + trainMSE.get(trainMSE.size() - 1));
		//System.out.println("train mse along the way: " + trainMSE.toString());
		System.out.println("final validation mse: " + testMSE.get(testMSE.size() - 1));
		//System.out.println("validation mse along the way: " + testMSE.toString());
		System.out.println("final validation accuracy: " + accuracies.get(accuracies.size() - 1));
		//System.out.println("validation accuracy along the way: " + accuracies.toString());
		System.out.println("Total epochs: " + cEpochs + "\n\n");
		
	}
	
	public double calculateAccuracy(Matrix features, Matrix labels) throws Exception{
		int correctCount = 0;
		double[] prediction = new double[1];
		for(int i = 0; i < features.rows(); i++){
			double[] feat = new double[features.cols()];
			feat = features.row(i);
								
			int targ = (int)labels.get(i, 0);
			predict(feat, prediction);
			int pred = (int)prediction[0];
								
			if(pred == targ)
				correctCount++;
		}
		double currentAccuracy = (double)correctCount / features.rows();
		
		return currentAccuracy;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		double iFinalIndex = -1;
		for(int i = 0; i < rgLayers.size(); i++){
			if(i == 0) 		// first hidden layer
			{	
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.length; k++){
						net += features[k]*rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1+Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
			else if(i < rgLayers.size() - 1) 		// other hidden layers
			{	
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < rgLayers.get(i - 1).size(); k++){
						net += rgLayers.get(i - 1).get(k).getOutput() * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1+Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
			else	// for output layer
			{	
				double answer = -1;
				int index = -1;
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias * rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < rgLayers.get(i-1).size(); k++){
						net += rgLayers.get(i - 1).get(k).getOutput() * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
					if(output > answer){
						answer = output;
						index = j;
					}
				}
				iFinalIndex = index;
			}
		}
		labels[0] = iFinalIndex;
		
	}
	public double predict(double[] features, int target) throws Exception 
	{
		double mse = 0;
		
		for(int i = 0; i < rgLayers.size(); i++){
			if(i == 0) {	// first hidden layer	
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.length; k++){
						net += features[k] * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
			else if(i < rgLayers.size()-1) {		// other hidden layers
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < rgLayers.get(i - 1).size(); k++){
						net += rgLayers.get(i - 1).get(k).getOutput() * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
				}
			}
			else{	// for the output layer
				for(int j = 0; j < rgLayers.get(i).size(); j++){
					double answer = -1;
					if(target == j)	//should be outputting true
						answer = 1;
					else			//should be outputting false
						answer = 0;
					double net = bias*rgLayers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < rgLayers.get(i - 1).size(); k++){
						net += rgLayers.get(i - 1).get(k).getOutput() * rgLayers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					rgLayers.get(i).get(j).setOutput(output);
					mse += Math.pow((answer-output), 2)/2.0;
					assert(answer != -1);
				}
			}
		}		
		return mse;
	}
	
	private class Node
	{
		
		private double[] rgWeight;
		private double[] rgDWeightChange;
		private int numWeight;
		private double output;
		private double delta;
		
		public Node(int numWeight){
			this.setOutput(0);
			this.setDelta(0);
			this.rgWeight = new double[numWeight];
			this.rgDWeightChange = new double[numWeight];
			this.numWeight = numWeight;
			initWeights(); 		// when created, the node initializes its weights
		}

		public void initWeights(){
			for(int i = 0; i < rgWeight.length; i++){
				rgWeight[i] = getRandomDouble();
			}
		}
		
		public double getRandomDouble()	{ //returns a random number with mean of zero
			double min = -1 / Math.sqrt(numWeight);
			double max = 1 / Math.sqrt(numWeight);
			
			while(true){
				double output = random.nextGaussian();
				if(output <= max && output >= min && output != 0){
					return output;
				}
			}
		}
		
		public void setWeightElement(int index, double change){
			this.rgWeight[index] = change;
		}
		
		public double getWeightElement(int index){
			return rgWeight[index];
		}
		
		public double getOutput() {
			return output;
		}
		
		public void setOutput(double output){
			this.output = output;
		}
		
		public void setWeightChangeElement(int index, double change){
			this.rgDWeightChange[index] = change;
		}
		
		public double getWeightChangeElement(int index){
			return rgDWeightChange[index];
		}
		
		public int getNumWeight(){
			return numWeight;
		}

		public double getDelta() {
			return delta;
		}
		
		public void setDelta(double delta) {
			this.delta = delta;
		}
	}
}