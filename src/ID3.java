import java.util.Random;

public class ID3 extends SupervisedLearner{

	private boolean prune = true;
	private final boolean BIN_CONTINUOUS = true;
	private ID3Tree tree;
	private Random rand;
	private final double TRAINING_PERCENT = 0.80;
	private final int C_BINS = 10;
	private double[] intervalChange;
	
	public ID3(Random rand){
		this.rand = rand;
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {
		Matrix mxValidationFeatures = null;
		Matrix mxValidationLabels = null;
		if(this.BIN_CONTINUOUS){
			intervalChange = new double[features.cols()];
		}
		this.repairMatrix(features);
		this.repairMatrix(labels);
		features.shuffle(rand, labels);
		if(prune){
			int trainingSize = (int)(TRAINING_PERCENT * features.rows());
			Matrix mxTrainFeatures = new Matrix(features, 0, 0, trainingSize, features.cols());
			Matrix mxTrainLabels = new Matrix(labels, 0, 0, trainingSize, 1);
			mxValidationFeatures = new Matrix(features, trainingSize, 0, features.rows()-trainingSize, features.cols());
			mxValidationLabels = new Matrix(labels, trainingSize, 0, features.rows()-trainingSize, 1);
			tree = new ID3Tree(mxTrainFeatures, mxTrainLabels);
		}
		else{
			tree = new ID3Tree(features, labels);
		}
		DecisionTreeNode root = tree.getRoot();
		tree.rExpandTree(root);
		if(prune){
			tree.pruneTree(mxValidationFeatures, mxValidationLabels);
			//System.out.println("Nodes in tree (w/ pruning): " + tree.getNodeCount());
		}
		else{
			//System.out.println("Nodes in tree (no pruning): " + tree.getNodeCount());
		}
		//System.out.println(tree.toString());
	}

	public void predict(double[] features, double[] labels) throws Exception {
		for (int i = 0; i < features.length; ++i){
			if(this.BIN_CONTINUOUS && intervalChange[i]!=1){
				features[i] = (int)(features[i]/intervalChange[i]);
			}
		}
		labels[0] = tree.predict(features);
	}
	
	private void repairMatrix(Matrix toRepair){
		
		//replace missing data
		for(int j = 0; j < toRepair.cols(); ++j){
			for(int i = 0; i < toRepair.rows(); ++i){
				if(toRepair.get(i, j) == Matrix.MISSING){
					toRepair.set(i, j, toRepair.mostCommonValue(j));
					//System.out.println("Repaired (" + i + ", " + j + ") with current value of " + toRepair.get(i, j));
				}
			}
		}
		
		//bin continuous data
		if(BIN_CONTINUOUS){
			for(int j = 0; j < toRepair.cols(); ++j){
				intervalChange[j] = 1;
				if(toRepair.valueCount(j)==0){	//continuous
					//System.out.println("Found continuous data");
					double maxValue = Double.MIN_VALUE;
					double minValue = Double.MAX_VALUE;
					for(int i = 0; i < toRepair.rows(); ++i){
						double tempValue = toRepair.get(i, j);
						maxValue = Math.max(tempValue, maxValue);
						minValue = Math.min(tempValue, minValue);
					}
					intervalChange [j] = (maxValue-minValue)/C_BINS;
					//System.out.println("IntervalChange: " + intervalChange);
					for(int i = 0; i < toRepair.rows(); ++i){
						//System.out.println("Previous value: " + toRepair.get(i, j));
						toRepair.set(i, j, (int)(toRepair.get(i, j)/intervalChange[j]));
						//System.out.println("New bin: " + toRepair.get(i, j));
					}
				}
			}
		}
	}

}
