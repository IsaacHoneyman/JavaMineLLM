package nn;

import java.util.List;

public abstract class Layer {
    public abstract Tensor forward(Tensor input);
    public abstract List<Tensor> parameters();
}
