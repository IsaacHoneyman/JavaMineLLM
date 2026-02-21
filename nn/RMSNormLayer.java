package nn;

import java.util.List;
import linalg.Matrix;

public class RMSNormLayer extends Layer {
    public Tensor gamma;

    public RMSNormLayer(int dim){
        Matrix gData = new Matrix(1, dim);
        for (int i = 0; i < dim; i++) gData.set(0, i, 1.0f);
        this.gamma = new Tensor(gData);
    }

    @Override
    public Tensor forward(Tensor input){
        Matrix norm = input.data.rmsNorm();
        Matrix resultData = Matrix.hadamard(norm, gamma.data); // can do hadamard as 1 col
        Tensor out = new Tensor(resultData);
        out.parents = new Tensor[] { input, gamma };

        out.backward = () -> {
            Matrix dGamma = new Matrix(gamma.data.rows, gamma.data.columns);
            for (int y = 0; y < input.data.rows; y++){
                for (int x = 0; x < input.data.columns; x++){
                    float val = dGamma.get(0, x) + (out.grad.get(y, x) * norm.get(y, x));
                    dGamma.set(0, x, val);
                }
            }
            gamma.grad = Matrix.add(gamma.grad, dGamma);

            Matrix weightedGrad = Matrix.hadamard(out.grad, gamma.data);
            input.grad = Matrix.add(input.grad, input.data.rmsNormBackward(weightedGrad));
        };

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(gamma);
    }
}
