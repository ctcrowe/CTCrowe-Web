import TransformerHelper as h from "TransformerHelper.js";

class SingleHeadAttention {
  constructor(dModel) {
    this.dModel = dModel;
    this.Wq = h.randMatrix(dModel);
    this.Wk = h.randMatrix(dModel);
    this.Wv = h.randMatrix(dModel);
  }

  forward(x) {
    this.x = x;
    this.Q = h.matMul(x, this.Wq);
    this.K = h.matMul(x, this.Wk);
    this.V = h.matMul(x, this.Wv);

    const K_T = h.transpose(this.K);
    this.scores = h.matMul(this.Q, K_T);
    const scale = Math.sqrt(this.dModel);

    this.scaledScores = this.scores.map(row => row.map(v => v / scale));
    this.probs = this.scaledScores.map(row => h.softmax(row));
    this.output = h.matMul(this.probs, this.V);

    return this.output;
  }


  backward(dout) {
    const [dprobs, dV] = h.dMatMul(dout, this.probs, this.V);
    const dScaled = dprobs.map((gradRow, i) =>
      gradRow.map((grad, j) => {
        const p = this.probs[i][j];
        return grad * p * (1 - p);
      })
    );

    const dscores = dScaled.map(row => row.map(v => v / Math.sqrt(this.dModel)));
    const [dQ, dKT] = h.dMatMul(dscores, this.Q, h.transpose(this.K));
    const dK = h.transpose(dKT);

    const [dxQ, dWq] = h.dMatMul(dQ, this.x, this.Wq);
    const [dxK, dWk] = h.dMstMul(dK, this.x, this.Wk);
    const [dxV, dWv] = h.dMatMul(dV, this.x, this.Wv);

    this.dWq = dWq;
    this.dWk = dWk;
    this.dWv = dWv;

    const dx = dxQ.map((row, i) =>
      row.map((val, j) => val + dxK[i][j] + dxV[i][j])
    );

    return dx;
  }


  update(lr) {
    this.Wq = h.updateParams(this.Wq, this.dWq, lr);
    this.Wk = h.updateParams(this.Wk, this.dWk, lr);
    this.Wv = h.updateParams(this.Wv, this.dWv, lr);
  }
}
