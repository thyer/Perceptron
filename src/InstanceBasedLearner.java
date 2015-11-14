import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

public class InstanceBasedLearner extends SupervisedLearner {
	private Random rand;
	private Matrix mxFeatures;
	private Matrix mxLabels;
	private double[] colAverage = null;
	private final int K = 9;
	private final boolean DISTANCE_WEIGHTING = true;
	private final boolean REGRESSION = false;


	public InstanceBasedLearner(Random rand) {
		this.rand = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		int rowsToKeep = (int) (1 * features.rows());
		features.shuffle(this.rand, labels);
		this.mxFeatures = new Matrix(features, 0, 0, rowsToKeep, features.cols());
		this.mxLabels = new Matrix(labels, 0, 0, rowsToKeep, labels.cols());
		this.repairMatrix(this.mxFeatures);
		//this.reduceRows(this.mxFeatures, this.mxLabels);
	}

	private void reduceRows(Matrix features, Matrix labels) throws Exception {
		System.out.println("At beginning, there are " + features.rows() + " instances");
		double validationSize = 0.1;
		int validationRows = (int) (features.rows() * validationSize);
		Matrix mxValidationFeatures = new Matrix(features, 0, 0, validationRows, features.cols());
		Matrix mxToPruneFeatures = new Matrix(features, validationRows, 0, features.rows()-validationRows, features.cols());
		Matrix mxValidationLabels = new Matrix(labels, 0, 0, validationRows, labels.cols());
		Matrix mxToPruneLabels = new Matrix(labels, validationRows, 0, labels.rows()-validationRows, labels.cols());
		for(int i = 0; i < mxToPruneFeatures.rows(); ++i){
			this.mxFeatures = mxToPruneFeatures;
			this.mxLabels = mxToPruneLabels;
			if(!this.assessInstance(i, mxValidationFeatures, mxValidationLabels)){
				Matrix tempFeatures = new Matrix(mxToPruneFeatures, 0, 0, i, mxToPruneFeatures.cols());
				Matrix tempLabels = new Matrix(mxToPruneLabels, 0, 0, i, 1);
				if(i!=mxToPruneFeatures.rows()-1){
					tempFeatures.add(mxToPruneFeatures, i+1, 0, mxToPruneFeatures.rows()-i-1);
					tempLabels.add(mxToPruneLabels, i+1, 0, mxToPruneFeatures.rows()-i-1);
				}
				i--;
				mxToPruneFeatures = tempFeatures;
				mxToPruneLabels = tempLabels;
			}
		}
		this.mxFeatures = mxToPruneFeatures;
		this.mxLabels = mxToPruneLabels;
		this.mxFeatures.add(mxValidationFeatures, 0, 0, mxValidationFeatures.rows());
		this.mxLabels.add(mxValidationLabels, 0, 0, mxValidationLabels.rows());
		System.out.println("At end, there are " + mxFeatures.rows() + " instances");
		
	}

	private boolean assessInstance(int instance, Matrix mxValidationFeatures, Matrix mxValidationLabels) throws Exception {
		//tempFeatures,tempLabels are the vectors without the given instance
		Matrix tempFeatures = new Matrix(mxFeatures, 0, 0, instance, mxFeatures.cols());
		Matrix tempLabels = new Matrix(mxLabels, 0, 0, instance, 1);
		if(instance!=mxFeatures.rows()-1){
			tempFeatures.add(mxFeatures, instance+1, 0, mxFeatures.rows()-instance-1);
			tempLabels.add(mxLabels, instance+1, 0, mxFeatures.rows()-instance-1);
		}
		
		//get accuracy with the instance
		double accuracy1 = 0;
		for(int i = 0; i < mxValidationFeatures.rows(); ++i){
			double prediction[] = {0.0};
			this.predict(mxValidationFeatures.row(i), prediction);
			if(prediction[0] == mxValidationLabels.row(i)[0]){
				accuracy1 += 1.0/mxLabels.rows();
			}
		}
		
		//get accuracy without the instance
		double accuracy2 = 0;
		this.mxFeatures = tempFeatures;
		this.mxLabels = tempLabels;
		for(int i = 0; i < mxValidationFeatures.rows(); ++i){
			double prediction[] = {0.0};
			this.predict(mxValidationFeatures.row(i), prediction);
			if(prediction[0] == mxValidationLabels.row(i)[0]){
				accuracy2 += 1.0/mxLabels.rows();
			}
		}
		//System.out.println("Accuracy with: " + accuracy1 + ", accuracy without: " + accuracy2);
		return accuracy1 >= accuracy2;
	}

	private void repairMatrix(Matrix mx) {
		colAverage = new double[mx.cols()];
		int repaired = 0;
		for(int j = 0; j < mx.cols(); ++j){
			colAverage[j] = mx.columnMean(j);
			for(int i = 0; i < mx.rows(); ++i){
				if(mx.get(i, j) == Matrix.MISSING){
					mx.set(i, j, colAverage[j]);
					repaired++;
				}
			}
		}
		System.out.println("Repaired " + repaired + " cells");	
		System.out.println("colAverage: ");
		for (double d : colAverage){
			System.out.println("\t" + d);
		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		//calculate and order distances to each instance
		for(int j = 0; j < features.length; ++j){
			if(colAverage != null && features[j] == Matrix.MISSING){
				features[j] = this.colAverage[j];
			}
		}
		PriorityQueue<Point> bestK = new PriorityQueue<Point>();
		for(int i = 0; i < mxFeatures.rows(); ++i){
			double dist = calcDistance(features, mxFeatures.row(i));
			bestK.add(new Point(mxLabels.row(i)[0], dist));
		}

		//select best K instances for voting
		Map<Double, Double> votingMap = new HashMap<Double, Double>();
		for(int i = 0; i < K; ++i){
			Point p = bestK.peek();
			bestK.remove();
			
			if(votingMap.containsKey(p.label)){
				double total = votingMap.get(p.label);
				votingMap.remove(p.label);
				votingMap.put(p.label, total + this.calcVote(p));
			}
			else{
				votingMap.put(p.label, this.calcVote(p));
			}
		}
		
		//finally, vote
		double bestLabel = 0.0;
		if(REGRESSION){
			double totalWeights = 0.0;
			for(double key : votingMap.keySet()){
				bestLabel += key * votingMap.get(key);
				totalWeights += votingMap.get(key);
			}
			labels[0] = bestLabel/totalWeights;
		}
		else{
			double bestLabelTotal = 0.0;
			for(double key : votingMap.keySet()){
				double total = votingMap.get(key);
				if(total > bestLabelTotal){
					bestLabel = key;
					bestLabelTotal = total;
				}
			}
			labels[0] = bestLabel;
		}
		

	}
	
	private double calcVote(Point p) {
		if(this.DISTANCE_WEIGHTING == true){
			return 1.0/Math.pow(p.distance + .0000001, 2.0);
		}
		else{
			return 1;
		}
	}

	public double calcDistance(double[] point1, double[] point2){
		if(point1.length != point2.length){
			assert(false);
			return -1.0;
		}
		else{
			double output = 0;
			for(int i = 0; i < point1.length; ++i){
				output += Math.pow(point1[i] - point2[i], 2.0);
			}
			return Math.sqrt(output);
		}
	}
	
	@SuppressWarnings("rawtypes")
	private class Point implements Comparable{
		public double label;
		public double distance;
		
		public Point(double label, double distance){
			this.label = label;
			this.distance = distance;
		}

		@Override
		public String toString(){
			return "Label: " + label + ", distance: " + distance;
		}

		@Override
		public int compareTo(Object o) {
			if(o.getClass().equals(this.getClass())){
				Point other = (Point) o;
				return this.distance > other.distance ? 1 : -1;
			}
			return 0;
		}
	}
}
