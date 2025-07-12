import TransformerHelper as helper from "TransformerHelper.js";

class SingleHeadAttention {
  constructor(dModel) {
    this.dModel = dModel;
    this.Wq = helper.randMatrix(dModel);
    this.Wk = helper.randMatrix(dModel);
    this.Wv = helper.randMatrix(dModel);
  }

  forward(x) {
    this.x = x;
    this.Q = helper.matMul(x, this.Wq);
    this.K = helper.matMul(x, this.Wk);
    this.V = helper.matMul(x, this.Wv);

    const K_T = helper.transpose(this.K);
    this.scores = helper.matMul(this.Q, K_T);
    const scale = Math.sqrt(this.dModel);

    this.scaledScores = this.scores.map(row => row.map(v => v / scale));
    this.probs = this.scaledScores.map(row => helper.softmax(row));
    this.output = helper.matMul(this.probs, this.V);

    return this.output;
  }


  backward(dout) {
    const [dprobs, dV] = helper.matMulBackward(dout, this.probs, this.V);
    const dScaled = dprobs.map((gradRow, i) =>
      gradRow.map((grad, j) => {
        const p = this.probs[i][j];
        return grad * p * (1 - p);
      })
    );

    const dscores = dScaled.map(row => row.map(v => v / Math.sqrt(this.dModel)));
    const [dQ, dKT] = helper.matMulBackward(dscores, this.Q, helper.transpose(this.K));
    const dK = helper.transpose(dKT);

    const [dxQ, dWq] = helper.matMulBackward(dQ, this.x, this.Wq);
    const [dxK, dWk] = helper.matMulBackward(dK, this.x, this.Wk);
    const [dxV, dWv] = helper.matMulBackward(dV, this.x, this.Wv);

    this.dWq = dWq;
    this.dWk = dWk;
    this.dWv = dWv;

    const dx = dxQ.map((row, i) =>
      row.map((val, j) => val + dxK[i][j] + dxV[i][j])
    );

    return dx;
  }


  update(lr) {
    this.Wq = helper.updateParams(this.Wq, this.dWq, lr);
    this.Wk = helper.updateParams(this.Wk, this.dWk, lr);
    this.Wv = helper.updateParams(this.Wv, this.dWv, lr);
  }
}