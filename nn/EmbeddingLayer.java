package nn;

import java.util.List;
import linalg.Matrix;

public class EmbeddingLayer extends Layer {
    public Tensor weights;

    public EmbeddingLayer(int vocabSize, int vocabDim)
    {
        Matrix wData = new Matrix(vocabSize, vocabDim);
        wData.randomise(-0.1f, 0.1f);
        weights = new Tensor(wData);
    }

    @Override
    public Tensor forward(Tensor input){
        int rows = input.data.rows;
        int cols = weights.data.columns;
        Matrix resultData = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++){
            int wordId = (int) input.data.get(i, 0);
            for (int j = 0; j < cols; j++){
                resultData.set(i, j, weights.data.get(wordId, j));
            }
        }

        Tensor out = new Tensor(resultData);
        out.parents = new Tensor[]{weights, input};

        out.backward = () -> {
            for (int i = 0; i < rows; i++){
                int wordId = (int) input.data.get(i, 0);
                for (int j = 0; j < cols; j++){
                    weights.grad.set(wordId, j, weights.grad.get(wordId, j) + out.grad.get(i, j));
                }
            }
        };

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weights);
    }
}
