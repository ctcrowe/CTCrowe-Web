import TransformerHelper as h from "TransformerHelper.js";
import "SingleHeadAttention.js";
import "FeedForward.js";

class TransformerLayer {
  constructor(dModel, dHidden, dOut) {
    this.attn = new SingleHeadAttention(dModel);
    this.ff = new FeedForward(dModel, dHidden, dOut);
  }

  forward(x) {
    this.x = x;
    const attnOut = this.attn.forward(x);
    const ffOut = this.ff.forward(attnOut);
    this.out = ffOut.map(row => softmax(row));
    return this.out;
  }

  backward(yTrue) {
    const dLoss = h.dSoftmaxCrossEntropy(this.out, yTrue);
    const dFF = this.ff.backward(dLoss);
    const dAttn = this.attn.backward(dFF);
    return dAttn; // dx (not used here, but useful if chaining)
  }

  update(lr) {
    this.attn.update(lr);
    this.ff.update(lr);
  }

  predict(x) {
    const out = this.forward(x);
    return out.map(row => row.indexOf(Math.max(...row)));
  }
}
