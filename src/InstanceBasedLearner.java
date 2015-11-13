import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

public class InstanceBasedLearner extends SupervisedLearner {
	private Random rand;
	private Matrix mxFeatures;
	private Matrix mxLabels;
	private final int K = 5;
	private final boolean DISTANCE_WEIGHTING = true;
	private final boolean REGRESSION = false;


	public InstanceBasedLearner(Random rand) {
		this.rand = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		this.mxFeatures = features;
		this.mxLabels = labels;
		this.mxFeatures.shuffle(this.rand, this.mxLabels);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		//calculate and order distances to each instance
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
