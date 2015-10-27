import java.util.ArrayList;

public class MatrixSplitter {
	private Matrix basefeatures = null;
	private Matrix baselabels = null;
	private int splitIndex = -1;
	ArrayList<Matrix> splitFeatures = null;
	ArrayList<Matrix> splitLabels = null;

	public MatrixSplitter(Matrix features, Matrix labels, int splitIndex) {
		this.basefeatures = features;
		this.baselabels = labels;
		this.splitIndex = splitIndex;
		this.performSplit();
	}

	public Matrix[] getSplitFeatures() {
		if(splitFeatures == null || splitFeatures.size() > 0){
			return (Matrix[])splitFeatures.toArray();
		}
		else return null;
	}

	public Matrix[] getSplitLabels() {
		if(splitLabels == null || splitLabels.size() > 0){
			return (Matrix[])splitLabels.toArray();
		}
		else return null;
	}
	
	//performs the split
	private void performSplit(){
		splitFeatures = new ArrayList<Matrix>();
		splitLabels = new ArrayList<Matrix>();
		
		//first, find out how many possible splits are in this feature
	}

}
