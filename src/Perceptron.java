import java.io.FileNotFoundException;

public class Perceptron {
	public static void main(String[] args){
		Matrix matrix = new Matrix();
		try {
			matrix.loadArff("C:\\Users\\Trent\\Documents\\GitHub\\Perceptron\\src\\votingMissingValuesReplaced.arff");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Hooray!");
	}
}
