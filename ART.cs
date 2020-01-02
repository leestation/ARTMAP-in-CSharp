using System;
using System.Collections.Generic;
using System.IO;

public class ART
{
	public ART() { }

	// static methods
	public static string DArray2String(Double[] array)
	{
		string result = "";
		for(int i=0; i<array.Length; i++)
		{
			result += array[i].ToString();
		}
		return result;
	}

	public static void CompCoding(Double[] I, Double[] x)
	{
		//if (x.Length != I.Length * 2) x = new double[I.Length * 2];
		if (x.Length != I.Length * 2) Array.Resize<double>(ref x, I.Length * 2);

		for (int i=0; i<I.Length; i++)
		{
			if (I[i] < 0) I[i] = 0;
			else if (I[i] > 1) I[i] = 1;

			x[i] = I[i];
			x[I.Length + i] = 1 - I[i];
		}
	}

	public static bool ArrayMin(double[] x, double[] w, double[] result)
	{
		if (x.Length != w.Length) return true; // abnormal case, return 1
		if (x.Length != result.Length) return true;

		for(int i=0; i<result.Length; i++)
		{
			result[i] = Math.Min(x[i], w[i]);
		}

		return false;
	}

	public static double OneNorm(double[] x)
	{
		double result = 0;
		for (int i = 0; i < x.Length; i++) result += x[i];
		return result;
	}

	public static double ChoiceFunc(double[] x, double[] w)
	{
		if (x.Length != w.Length) return 1; // abnormal case, return 1

		double result = 0;
		double alpha = 0.00001;
		double[] tempArray = new double[x.Length];
		if (ArrayMin(x, w, tempArray)) return 0; // abnormal case
		result = OneNorm(tempArray) + (1.0 - alpha) * (Convert.ToDouble(x.Length) - OneNorm(w));

		return result;
	}

	public static bool GetActVal(double[] x, List<double[]> w, double[] result)
	{
		if (w.Count != result.Length) return true; // abnormal case, return 1

		for (int i=0; i<result.Length; i++)
		{
			result[i] = ChoiceFunc(x, w[i]);
		}

		return false;
	}

	public static int GetMaxIndex(double[] T)
	{
		double maxVal = 0;
		int maxIndex = -1;
		for (int i=0; i<T.Length; i++)
		{
			if (T[i] > maxVal)
			{
				maxVal = T[i];
				maxIndex = i;
			}
		}

		return maxIndex;
		// if maxIndex == -1, every value is less than or equal to zero. need to add a new category
	}


	public static int ResonanceCheck(double[] x, double[] w, double vigilParam)
	{
		// return 1 (satisfied), 0 (unsatisfied), -1 (abnormal case)

		double[] tempArray = new double[x.Length];
		if (ArrayMin(x, w, tempArray)) return -1; // abnormal case

		if ( (2.0 * OneNorm(tempArray)) >= (vigilParam * Convert.ToDouble(x.Length)) ) return 1; // resonance condition satisfied

		return 0; // resonance condition unsatisfied
	}

	public static int TemplateMatching(double[] x, List<double[]> w, double vigilParam) // only for Fuzzy ART (not ARTMAP)
	{
		double[] T = new double[w.Count];
		if (ART.GetActVal(x, w, T)) return -1; // abnormal case

		while(true)
		{
			int J = ART.GetMaxIndex(T);
			if (J < 0) return w.Count; // every value is less than or equal to zero. need to add a new category

			int resonanceCheck = ResonanceCheck(x, w[J], vigilParam);

			if (resonanceCheck == -1) return -1; // abnormal case

			else if (resonanceCheck == 1) // template matched
			{
				return J; 
			}
			
			else if (resonanceCheck == 0) // template unmatched
			{
				T[J] = 0;
			}
		};
	}

	public static int TemplateMatching(double[] x, List<double[]> w, double vigilParam, List<int> l, int labelIndex)
	{
		// if label index is -1, it means the model doesn't do learning but only testing

		double variableVigPram = 0.1; // the lower, the more efficient for ARTMAP
		if (labelIndex == -1) variableVigPram = vigilParam;

		double[] T = new double[w.Count];
		if (ART.GetActVal(x, w, T)) return -1; // abnormal case

		while (true)
		{
			int J = ART.GetMaxIndex(T);
			if (labelIndex == -1) return J; // no need to check the resonance condition in the testing phase.
			if (J < 0) return w.Count; // every value is less than or equal to zero. need to add a new category

			int resonanceCheck = ResonanceCheck(x, w[J], variableVigPram);

			if (resonanceCheck == -1) return -1; // abnormal case

			else if (resonanceCheck == 1) // template matched
			{
				if (l[J] == labelIndex)
				{
					return J;
				}
				else // classification incorrect case
				{
					double[] tempArray = new double[x.Length];
					if (ArrayMin(x, w[J], tempArray)) return -1; // abnormal case
					variableVigPram = 2.0 * OneNorm(tempArray) / x.Length + 0.00001;
					T[J] = 0;
				}
			}

			else if (resonanceCheck == 0) // template unmatched
			{
				T[J] = 0;
			}
		};
	}

	public static int UpdateWeight(double[] x, List<double[]> w, int J, double learningRate)
	{
		double[] tempArray = new double[x.Length];
		if (ArrayMin(x, w[J], tempArray)) return -1; // abnormal case
		for (int i = 0; i < w[J].Length; i++)
			w[J][i] = (1.0 - learningRate) * w[J][i] + learningRate * (tempArray[i]);
		return 0;
	}

	public static int AddCategory(double[] x, List<double[]> w, double learningRate) // Fuzzy ART only
	{
		double[] newWeight = new double[x.Length];
		for (int i = 0; i < newWeight.Length; i++) newWeight[i] = 1.0;
		w.Add(newWeight);
		if (UpdateWeight(x, w, w.Count-1, learningRate) == -1) return -1; // abnormal case
		return 0;
	}

	public static int AddCategory(double[] x, List<double[]> w, List<int> l, int labelIndex, double learningRate) // for ARTMAP
	{
		double[] newWeight = new double[x.Length];
		for (int i = 0; i < newWeight.Length; i++) newWeight[i] = 1.0;
		w.Add(newWeight);
		l.Add(labelIndex);
		if (UpdateWeight(x, w, w.Count - 1, learningRate) == -1) return -1; // abnormal case
		return 0;
	}







	// member variables and functions (not static)

	public double vigilParam = 0.9;
	public double learningRate = 0.9;

	public List<double[]> w = new List<double[]>();
	public List<int> l = new List<int>();

	public int GetInputDimension()
	{
		// returns 0 if not determined
		if (w.Count == 0) return 0;
		// returns a positive integer number if it has been determined
		return w[0].Length/2; // any w[j] array has the same "EVEN" number
	}

	[Serializable]
	public struct ModelData
	{
		public List<double[]> w;
		public List<int> l;
	}

	public int SaveModel(string filePath)
	{
		ModelData data = new ModelData();

		data.w = new List<double[]>();
		for (int i = 0; i < w.Count; i++)
			data.w.Add(w[i]);
		data.l = new List<int>();
		for (int i = 0; i < l.Count; i++)
			data.l.Add(l[i]);

		WriteToBinaryFile<ModelData>(filePath, data);

		// example)
		// Write the list of salesman objects to file.
		//WriteToXmlFile<List<salesman>>("C:\salesmen.txt", salesmanList);
		// Read the list of salesman objects from the file back into a variable.
		//List<salesman> salesmanList = ReadFromXmlFile<List<salesman>>("C:\salesmen.txt");

		return 0;

	}

	public int LoadModel(string filePath)
	{
		ModelData data = ReadFromBinaryFile<ModelData>(filePath);

		// clear the variables
		try
		{
			w.Clear();
			w.TrimExcess();
			l.Clear();
			l.TrimExcess();
		}
		finally { }

		w = new List<double[]>();
		for (int i = 0; i < data.w.Count; i++)
			w.Add(data.w[i]);
		l = new List<int>();
		for (int i = 0; i < data.l.Count; i++)
			l.Add(data.l[i]);

		// example)
		// Write the list of salesman objects to file.
		//WriteToXmlFile<List<salesman>>("C:\salesmen.txt", salesmanList);
		// Read the list of salesman objects from the file back into a variable.
		//List<salesman> salesmanList = ReadFromXmlFile<List<salesman>>("C:\salesmen.txt");

		return 0;
	}

	public int FuzzyART(double[] I)
	{
		double[] x = new double[I.Length * 2];
		ART.CompCoding(I, x);

		int J = ART.TemplateMatching(x, w, vigilParam, null, -1);

		if (J < 0) return -1;
		else if (J < w.Count) ART.UpdateWeight(x, w, J, learningRate);
		else ART.AddCategory(x, w, learningRate);

		return J;
	}



	

	public int ARTMAP(double[] I, int labelIndex, bool learningFlag)
	{
		// if label index is -1, it means ARTMAP behaves like a unsupervised learning
		// if learningFlag is false, it means ARTMAP only tests the input without learning

		// check if the input dimension is approperate
		if (I.Length != GetInputDimension()) return -1;

		double[] x = new double[I.Length * 2];
		ART.CompCoding(I, x);

		int J = ART.TemplateMatching(x, w, vigilParam, l, labelIndex);

		if (learningFlag)
		{
			if (J < 0) return -1;
			else if (J < w.Count) ART.UpdateWeight(x, w, J, learningRate);
			else ART.AddCategory(x, w, l, labelIndex, learningRate);
			return labelIndex;
		}
		else
		{
			if (J < 0) return -1;
			return l[J];
		}
	}



	~ART()
	{
		w.Clear();
		w.TrimExcess();
		l.Clear();
		l.TrimExcess();
	}


	// below functions are obtained from here: https://stackoverflow.com/questions/16352879/write-list-of-objects-to-a-file/22416929

	/// <summary>
	/// Writes the given object instance to a binary file.
	/// <para>Object type (and all child types) must be decorated with the [Serializable] attribute.</para>
	/// <para>To prevent a variable from being serialized, decorate it with the [NonSerialized] attribute; cannot be applied to properties.</para>
	/// </summary>
	/// <typeparam name="T">The type of object being written to the binary file.</typeparam>
	/// <param name="filePath">The file path to write the object instance to.</param>
	/// <param name="objectToWrite">The object instance to write to the XML file.</param>
	/// <param name="append">If false the file will be overwritten if it already exists. If true the contents will be appended to the file.</param>
	public static void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
	{
		using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
		{
			var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
			binaryFormatter.Serialize(stream, objectToWrite);
		}
	}

	/// <summary>
	/// Reads an object instance from a binary file.
	/// </summary>
	/// <typeparam name="T">The type of object to read from the XML.</typeparam>
	/// <param name="filePath">The file path to read the object instance from.</param>
	/// <returns>Returns a new instance of the object read from the binary file.</returns>
	public static T ReadFromBinaryFile<T>(string filePath)
	{
		using (Stream stream = File.Open(filePath, FileMode.Open))
		{
			var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
			return (T)binaryFormatter.Deserialize(stream);
		}
	}
}
