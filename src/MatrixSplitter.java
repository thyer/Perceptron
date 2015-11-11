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
		if(splitFeatures != null && splitFeatures.size() > 0){
			return splitFeatures.toArray(new Matrix[splitFeatures.size()]);
		}
		else return null;
	}

	public Matrix[] getSplitLabels() {
		if(splitLabels != null && splitLabels.size() > 0){
			return splitLabels.toArray(new Matrix[splitLabels.size()]);
		}
		else return null;
	}
	
	private void performSplit(){
		this.splitFeatures = new ArrayList<Matrix>();
		this.splitLabels = new ArrayList<Matrix>();
		
		//first, find out how many possible splits are in this feature
		ArrayList<Double> categoriesSoFar = new ArrayList<Double>();
		ArrayList<Double> labelsSoFar = new ArrayList<Double>();
		for(int i = 0; i < this.basefeatures.rows(); ++i){
			double item = this.basefeatures.row(i)[this.splitIndex];
			double label = this.baselabels.row(i)[0];
			if (!categoriesSoFar.contains(item)){
				categoriesSoFar.add(item);
			}
			if(!labelsSoFar.contains(label)){
				labelsSoFar.add(label);
			}
		}
		//System.out.println("CATEGORIES ON SPLIT: " + categoriesSoFar.size());
		if(categoriesSoFar.size()<=1 || labelsSoFar.size() <= 1){
			return;	//we have no data or only one class in our matrix
		}
		else if (categoriesSoFar.size() == 1){	//only one category exists for that feature
			this.splitFeatures.add(new Matrix(this.basefeatures, 0, 0, this.basefeatures.rows(), this.basefeatures.cols()));
			this.splitLabels.add(new Matrix(this.baselabels, 0, 0, this.baselabels.rows(), this.baselabels.cols()));
		}
		else{
			for (Double category : categoriesSoFar){
				Matrix m = null;
				Matrix l = null;
				boolean initialized = false;
				for(int i = 0; i < this.basefeatures.rows(); ++i){
					if(this.basefeatures.row(i)[this.splitIndex] == category && !initialized){
						m = new Matrix(this.basefeatures, i, 0, 1, this.basefeatures.cols());
						l = new Matrix(this.baselabels, i, 0, 1, 1);
						initialized = true;
					}
					else if(this.basefeatures.row(i)[this.splitIndex] == category && initialized){
						try {
							m.add(this.basefeatures, i, 0, 1);
							l.add(this.baselabels, i, 0, 1);
						} catch (Exception e) {
							e.printStackTrace();
						}
						
					}
					else{
						continue;
					}
				}
				this.splitFeatures.add(m);
				this.splitLabels.add(l);
			}
		}
	}

}
