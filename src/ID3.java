import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

public class ID3 extends SupervisedLearner{

	private ID3Tree tree;
	private Random rand;
	
	public ID3(Random rand){
		this.rand = rand;
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {
		this.repairMatrix(features);
		this.repairMatrix(labels);
		features.shuffle(rand, labels);
		
		tree = new ID3Tree(features, labels);
		DecisionTreeNode root = tree.getRoot();
		tree.rExpandTree(root);
		tree.toString();
	}

	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = tree.predict(features);
	}
	
	private void repairMatrix(Matrix toRepair){
		for(int j = 0; j < toRepair.cols(); ++j){
			for(int i = 0; i < toRepair.rows(); ++i){
				if(toRepair.get(i, j) == Matrix.MISSING){
					toRepair.set(i, j, toRepair.mostCommonValue(j));
					//System.out.println("Repaired (" + i + ", " + j + ") with current value of " + toRepair.get(i, j));
				}
			}
		}
	}

}
