package nn;

import java.util.ArrayList;
import java.util.List;
import linalg.Matrix;

public class LinearLayer extends Layer {
    public Tensor weights;
    public Tensor bias;

    public LinearLayer(int inFeatures, int outFeatures) {
        Matrix wData = new Matrix(inFeatures, outFeatures);
        Matrix bData = new Matrix(1, outFeatures);

        float limit = (float) Math.sqrt(2.0 / inFeatures);
        wData.randomise(-limit, limit);
        bData.zero();

        this.weights = new Tensor(wData);
        this.bias = new Tensor(bData);
    }

    @Override
    public Tensor forward(Tensor x){
        return x.dot(weights).add(bias);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.add(weights);
        params.add(bias);
        return params;
    }
}
