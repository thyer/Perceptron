import java.util.Random;

public class Perceptron extends SupervisedLearner{
	private Random r;
	private double[] m_weights;
	private double learningRate = .4;
	
	
	public Perceptron(Random rand){
		this.r = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		//validation
		if(labels.cols() != 1){
			throw new Exception("label matrix had " + labels.cols() + " columns in it.");
		}
		
		//instantiate the weight vector with weights and bias set to random values [0...1]
		m_weights = new double[features.cols() + 1];
		for (int i = 0; i < m_weights.length; ++i){
			m_weights[i] = r.nextDouble();
		}
		
		//begin training
		int epocsSinceLastImprovement = 0;
		int mostCorrectGuesses = 0;
		int epoc = 0;
		while(epocsSinceLastImprovement < 5){
			epoc++;
			
			//iterate through an epoch
			int correctGuessesThisEpoch = 0;
			for (int i = 0; i < features.rows(); ++i){
				double output = 0;
				for (int j = 0; j <= features.row(i).length; ++j){
					double input = (j == features.row(i).length) ? 1 : features.row(i)[j];
					output += input * m_weights[j];
				}
				if((output >= 0 && labels.row(i)[0] == 1) || (output < 0 && labels.row(i)[0] == 0)){
					correctGuessesThisEpoch++;
				}
				else {
					//adjust weights
					if(output != 0){
						output /= Math.abs(output); // standardizes to 1, -1
					}
					for (int j = 0; j <= features.row(i).length; ++j){
						double input = j == features.row(i).length ? 1 : features.row(i)[j];
						double delta_weight = learningRate * (labels.row(i)[0] - output) * input;
						m_weights[j] += delta_weight;
					}
				}	
			}
			if(correctGuessesThisEpoch > mostCorrectGuesses){
				epocsSinceLastImprovement = 0;
				mostCorrectGuesses = correctGuessesThisEpoch;
			}
			else{
				epocsSinceLastImprovement++;
			}
		}

		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double total = 0;
		for(int i = 0; i < features.length; ++i){
			total += features[i] * m_weights[i];
		}
		labels [0] = (total + m_weights[m_weights.length-1] > 0) ? 1 : 0;
	}
}
