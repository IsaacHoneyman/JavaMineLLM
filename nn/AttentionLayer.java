package nn;

import java.util.ArrayList;
import java.util.List;
import linalg.UtilMath;

public class AttentionLayer extends Layer {
    public final LinearLayer query;
    public final LinearLayer key;
    public final LinearLayer value;
    public final LinearLayer output;

    private final float scale;

    public AttentionLayer(int dim){
        this.query = new LinearLayer(dim, dim);
        this.key = new LinearLayer(dim, dim);
        this.value = new LinearLayer(dim, dim);
        this.output = new LinearLayer(dim, dim);
    
        this.scale = (float) (1.0 / Math.sqrt(dim));
    }

    @Override
    public Tensor forward(Tensor x)
    {
        Tensor q = query.forward(x);
        Tensor k = key.forward(x);
        Tensor v = value.forward(x);

        Tensor scaledScore = q.dot(k.transpose()).multiply(scale);

        // apply a mask to prevent training on future words
        for (int i = 0; i < scaledScore.data.rows; i++) {
            for (int j = 0; j < scaledScore.data.columns; j++) {
                if (j > i) scaledScore.data.set(i, j, UtilMath.NEG_BILLON);
            }
        }

        return output.forward(scaledScore.softmax().dot(v));   
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(query.parameters());
        params.addAll(key.parameters());
        params.addAll(value.parameters());
        params.addAll(output.parameters());
        return params;
    }
}
