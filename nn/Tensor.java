package nn;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import linalg.Matrix;

public class Tensor {
    // fields are public for performance overhead

    public Matrix data;
    public Matrix grad; // gradients, (error signal)

    public Tensor[] parents;
    public Runnable backward;

    public Tensor(Matrix data) {
        this.data = data;
        this.grad = new Matrix(data.rows, data.columns);
        this.parents = new Tensor[0];
        this.backward = () -> {
        };
    }

    public Tensor(Matrix data, Tensor[] parents, Runnable backward) {
        this.data = data;
        this.grad = new Matrix(data.rows, data.columns);
        this.parents = parents.clone();
        this.backward = backward;
    }

    // --- Tensor Manipulation ---

    public static Tensor add(Tensor a, Tensor b) {
        Matrix resultData = Matrix.add(a.data, b.data);
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { a, b };

        result.backward = () -> {
            a.grad = Matrix.add(a.grad, result.grad);
            b.grad = Matrix.add(b.grad, result.grad);
        };

        return result;
    }

    public Tensor add(Tensor other) {
        return add(this, other);
    }

    public static Tensor dot(Tensor a, Tensor b) {
        Matrix resultData = Matrix.dot(a.data, b.data);
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { a, b };

        result.backward = () -> {
            a.grad = Matrix.add(a.grad, Matrix.dot(result.grad, b.data.transpose()));
            b.grad = Matrix.add(b.grad, Matrix.dot(a.data.transpose(), result.grad));
        };

        return result;
    }

    public Tensor dot(Tensor other) {
        return dot(this, other);
    }

    public Tensor multiply(float scalar) {
        Matrix resultData = data.multiply(scalar);
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { this };

        result.backward = () -> {
            this.grad = Matrix.add(this.grad, result.grad.multiply(scalar));
        };

        return result;
    }

    public Tensor transpose() {
        Matrix resultData = data.transpose();
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { this };

        result.backward = () -> {
            grad = Matrix.add(grad, result.grad.transpose());
        };

        return result;
    }

    public Tensor gelu() {
        Matrix resultData = data.gelu();
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { this };

        result.backward = () -> {
            grad = Matrix.add(grad, data.geluBackward(result.grad));
        };

        return result;
    }

    public Tensor softmax() {
        Matrix resultData = data.softmax();
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { this };

        result.backward = () -> {
            grad = Matrix.add(grad, resultData.softmaxBackward(result.grad));

        };

        return result;
    }

    public Tensor rmsNorm() {
        Matrix resultData = data.rmsNorm();
        Tensor result = new Tensor(resultData);
        result.parents = new Tensor[] { this };

        result.backward = () -> {
            grad = Matrix.add(grad, data.rmsNormBackward(result.grad));

        };

        return result;
    }

    // --- Tesnor Search ---

    /**
     * Triggers the full backpropagation process.
     * Call on final scalar Loss tensor
     */
    public void backwardprop() {
        List<Tensor> topoSort = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();

        buildTopo(this, visited, topoSort);
        this.grad.set(0, 0, 1); // dl / dl = 1.0, so insert into 1 x 1 loss matrix

        for (int i = topoSort.size() - 1; i >= 0; i--) {
            topoSort.get(i).backward.run();
        }
    }

    private void buildTopo(Tensor node, Set<Tensor> visited, List<Tensor> topoSort) {
        if (visited.contains(node))
            return;
        
        visited.add(node);
        for (Tensor p : node.parents) {
            buildTopo(p, visited, topoSort);
        }

        topoSort.add(node);
    }

}
