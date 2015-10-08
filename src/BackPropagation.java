import java.util.Random;
import java.util.ArrayList;

public class BackPropagation extends SupervisedLearner
{
	Random random;
	ArrayList<ArrayList<Node>> layers;	
	int bias = 1;
	
	public BackPropagation(Random random){
		this.random = random;
	}
	
	private void makeLayer(int cInputNodes, int cOutputNodes, int cLayers){
		layers = new ArrayList<ArrayList<Node>>(); 
		
		int[] cNodesPerLayer = new int[cLayers + 2];

		cNodesPerLayer[0] = cInputNodes;
		cNodesPerLayer[1] = 64; 		// TODO: specify the number of nodes for the first hidden layer
		cNodesPerLayer[2] = 64;			// TODO: specify the number of nodes for the second hidden layer
		cNodesPerLayer[3] = 64;
		cNodesPerLayer[4] = 64;	
		cNodesPerLayer[5] = cOutputNodes;		
		
		for(int i = 0; i <= cLayers; i++){
			ArrayList<Node> iLayer = new ArrayList<Node>();			
			for(int j = 0; j < cNodesPerLayer[i+1]; j++){
				Node tNode = new Node(cNodesPerLayer[i] + 1);
				iLayer.add(tNode);
			}
			layers.add(iLayer);
		}			
	}

	public void forward_path(Matrix features, int a){ // where a is row of matrix
		for(int i = 0; i < layers.size(); i++){
			if(i == 0){ 		// first hidden layer	
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias * layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.cols(); k++){
						net += features.get(a, k) * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1/(1+Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
			else{
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < layers.get(i).get(j).getNumWeight() - 1; k++){
						net += layers.get(i - 1).get(k).getOutput() * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
		}	
	}
	
	public double backward_path(Matrix features, Matrix labels, int a){
		double dLearningRate = 0.1;		// specify the learning rate
		double momentum = 0;
		double mse = 0;
		
		for(int i = layers.size()-1; i >= 0; i--){
			if(i == layers.size()-1){ 		// output layer
				for(int j = 0; j < layers.get(i).size(); j++){
					double output = layers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					int answer = -1;
					if(labels.get(a, 0) == j){
						answer = 1;
					}
					else{
						answer = 0;
					}
					double delta = (answer-output) * fnet;
					
					mse += 0.5 * Math.pow((answer-output), 2);
					
					layers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < layers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (dLearningRate * delta * bias) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						else
							change = (dLearningRate * delta * layers.get(i-1).get(k-1).getOutput()) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						layers.get(i).get(j).setWeightChangeElement(k, change);
					}
				}
			}

			else if(i > 0){			// hidden layer
				for(int j = 0; j < layers.get(i).size(); j++){
					double output = layers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					double sum = 0;
					for(int k = 0; k < layers.get(i + 1).size(); k++){
						sum += layers.get(i + 1).get(k).getDelta() * layers.get(i + 1).get(k).getWeightElement(j + 1);
					}
					double delta = fnet * sum;
					layers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < layers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (dLearningRate * delta * bias) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						else
							change = (dLearningRate * delta * layers.get(i - 1).get(k - 1).getOutput()) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						layers.get(i).get(j).setWeightChangeElement(k, change);
					}						
				}
			}
			else{			// first layer	
				for(int j = 0; j < layers.get(i).size(); j++){
					double output = layers.get(i).get(j).getOutput();
					double fnet = output * (1 - output);
					double sum = 0;
					for(int k = 0; k < layers.get(i + 1).size(); k++){
						sum += layers.get(i + 1).get(k).getDelta() * layers.get(i + 1).get(k).getWeightElement(j + 1);
					}
					double delta = fnet * sum;
					layers.get(i).get(j).setDelta(delta);
					for(int k = 0; k < layers.get(i).get(j).getNumWeight(); k++){
						double change = 0;
						if(k == 0) 		// for bias node output
							change = (dLearningRate * delta * bias) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						else
							change = (dLearningRate * delta * features.get(a, k - 1)) + (momentum * layers.get(i).get(j).get_weight_change_element(k));
						layers.get(i).get(j).setWeightChangeElement(k, change);
					}						
				}					
			}
		}
		return mse;
	}

	public void update_weights(){
		for(int i = 0; i < layers.size(); i++){
			for(int j = 0; j < layers.get(i).size(); j++){
				for(int k = 0; k < layers.get(i).get(j).numWeight; k++){
					double new_weight = layers.get(i).get(j).getWeightElement(k) + layers.get(i).get(j).get_weight_change_element(k);
					layers.get(i).get(j).setWeightElement(k, new_weight);
				}
			}
		}
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		int num_input = features.cols(); 
		int num_output = labels.valueCount(0);
		int num_layer = 4;		// specify the number of hidden layers
		makeLayer(num_input, num_output, num_layer);		// a function to set up the hidden layers
		
		int training_size = (int)(0.9 * features.rows());

		
		int counter = 0;
		boolean continue_loop = true;
		
		ArrayList<Double> train_mse = new ArrayList<>();
		ArrayList<Double> test_mse = new ArrayList<>();
		ArrayList<Double> accuracies = new ArrayList<>();
		
		do{	
			features.shuffle(random, labels);
			Matrix train_features = new Matrix(features, 0, 0, training_size, features.cols());
			Matrix train_labels = new Matrix(labels, 0, 0, training_size, 1);
			Matrix test_features = new Matrix(features, training_size, 0, features.rows()-training_size, features.cols());
			Matrix test_labels = new Matrix(labels, training_size, 0, features.rows()-training_size, 1);
			double mse = 0;
			for(int i = 0; i < train_features.rows(); i++)
			{
				forward_path(train_features, i);
				mse += backward_path(train_features, train_labels, i);
				update_weights();
			}
			mse = mse/train_features.rows();
			train_mse.add(mse);
			
			mse = 0;
			for(int i = 0; i < test_features.rows(); i++)
			{	
				forward_path(test_features, i);
				mse += backward_path(test_features, test_labels, i);
				update_weights();
			}
			mse = mse/test_features.rows();
			test_mse.add(mse);
			
			double current_accuracy = calculate_accuracy(test_features, test_labels);
			
			if(counter > 100)
			{
				double lowest = 1;
				for(int i = 0; i < 100; i++)
				{
					if(accuracies.get(accuracies.size() - 1 - i) < lowest)
						lowest = accuracies.get(accuracies.size() - 1 - i);
				}
				if(lowest > current_accuracy)
					continue_loop = false;
			}
			accuracies.add(current_accuracy);
			counter++;
		}while(continue_loop);
		
		System.out.println("train_mse: " + train_mse.get(train_mse.size() - 1));
		System.out.println("validation_mse: " + test_mse.get(test_mse.size() - 1));
		System.out.println("validation_accuracy: " + accuracies.get(accuracies.size() - 1));
		
	}
	
	public double calculate_accuracy(Matrix features, Matrix labels) throws Exception{
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
		double current_accuracy = (double)correctCount / features.rows();
		
		return current_accuracy;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		double final_index = -1;
		for(int i = 0; i < layers.size(); i++){
			if(i == 0) 		// first hidden layer
			{	
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.length; k++){
						net += features[k]*layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1+Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
			else if(i < layers.size() - 1) 		// other hidden layers
			{	
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < layers.get(i - 1).size(); k++){
						net += layers.get(i - 1).get(k).getOutput() * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1+Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
			else	// for output layer
			{	
				double answer = -1;
				int index = -1;
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias * layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < layers.get(i-1).size(); k++){
						net += layers.get(i - 1).get(k).getOutput() * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
					if(output > answer){
						answer = output;
						index = j;
					}
				}
				final_index = index;
			}
		}
		labels[0] = final_index;
		
	}
	public double predict(double[] features, int target) throws Exception 
	{
		double mse = 0;
		
		for(int i = 0; i < layers.size(); i++){
			if(i == 0) {	// first hidden layer	
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < features.length; k++){
						net += features[k] * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
			else if(i < layers.size()-1) {		// other hidden layers
				for(int j = 0; j < layers.get(i).size(); j++){
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < layers.get(i - 1).size(); k++){
						net += layers.get(i - 1).get(k).getOutput() * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
				}
			}
			else{	// for the output layer
				for(int j = 0; j < layers.get(i).size(); j++){
					double answer = -1;
					if(target == j)	//should be outputting true
						answer = 1;
					else			//should be outputting false
						answer = 0;
					double net = bias*layers.get(i).get(j).getWeightElement(0);
					for(int k = 0; k < layers.get(i - 1).size(); k++){
						net += layers.get(i - 1).get(k).getOutput() * layers.get(i).get(j).getWeightElement(k + 1);
					}
					double output = 1 / (1 + Math.exp(-net));
					layers.get(i).get(j).setOutput(output);
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
		
		public double getRandomDouble()	{ //returns a random number following gaussian distribution
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
		
		public double get_weight_change_element(int index){
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