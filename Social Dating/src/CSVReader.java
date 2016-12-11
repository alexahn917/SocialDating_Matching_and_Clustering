import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;

public class CSVReader {

	public static void main(String[] args) {
		String csvFile = "/Users/Alex/Documents/GitHub/SocialDating_Matching_and_Clustering/Social Dating/Data/Social_Dating_Full_Data.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		BufferedWriter bw = null;
		try {
			String savedFile = "/Users/Alex/Documents/GitHub/SocialDating_Matching_and_Clustering/Social Dating/Data/full_data";
			File file = new File(savedFile);
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			bw = new BufferedWriter(fw);

			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {
				String[] data = line.split(cvsSplitBy);
				String outputVal = "";
				for (int i = 0; i < data.length; i++) {
					if (i == 0) {
						outputVal = data[i];
					} 
					else if (!data[i].equals("") && !data[i].equals("NULL")) {
						outputVal = outputVal + " " + i +":" +data[i];
					}
				}
				outputVal = outputVal + "\n";
				bw.write(outputVal);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
					bw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}

}