package nn;

import java.util.ArrayList;
import java.util.List;

public class TransformerBlockLayer extends Layer {
    public final RMSNormLayer norma;
    public final RMSNormLayer normb;
    public final AttentionLayer attention;
    public final FeedForwardLayer feedForward;

    public TransformerBlockLayer(int dim, int hiddenDim){
        this.norma = new RMSNormLayer(dim);
        this.normb = new RMSNormLayer(dim);

        this.attention = new AttentionLayer(dim);
        this.feedForward = new FeedForwardLayer(dim, hiddenDim);
    }

    @Override
    public Tensor forward(Tensor x){
        x = x.add(attention.forward(norma.forward(x)));
        return x.add(feedForward.forward(normb.forward(x)));
    }

    @Override
    public List<Tensor> parameters(){
        List<Tensor> params = new ArrayList<>();
        params.addAll(norma.parameters());
        params.addAll(normb.parameters());
        params.addAll(attention.parameters());
        params.addAll(feedForward.parameters());
        return  params;
    }
}
