package nn;

import java.util.ArrayList;
import java.util.List;

public class FeedForwardLayer extends Layer {
    public final LinearLayer lla;
    public final LinearLayer llb;

    public FeedForwardLayer(int dim, int hiddenDim){
        this.lla = new LinearLayer(dim, hiddenDim);
        this.llb = new LinearLayer(hiddenDim, dim);
    }

    @Override
    public Tensor forward(Tensor x){
        return llb.forward(lla.forward(x).gelu());
    }

    @Override
    public List<Tensor> parameters(){
        List<Tensor> params = new ArrayList<>();
        params.addAll(lla.parameters());
        params.addAll(llb.parameters());
        return params;
    }
}
