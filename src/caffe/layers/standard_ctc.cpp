#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {
template <typename Dtype>
void ConnectionistTemporalClassificationLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom,top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void ConnectionistTemporalClassificationLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_GE(outer_num_ * inner_num_, bottom[1]->count());
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void ConnectionistTemporalClassificationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ConnectionistTemporalClassificationLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
//  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
//  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  
  // T should be rewritten
  T = static_cast<int>(bottom[2]->cpu_data()[0]); // sequence length (cf. t)
  //LOG(INFO) << "T: " << T;
  int dim = bottom[0]->count() / bottom[0]->num();
  //LOG(INFO) << "dim: " << dim;
  CHECK_GE(bottom[0]->num(), T);
  // Generate the real label, namely excluding the blanks
  vector<int> real_label;
  for (int t = 0; t < T; ++t) {
    int current_label = static_cast<int>(bottom_label[t]);
    if (current_label!=0)
      real_label.push_back(current_label);
  }
  int l_label = real_label.size(); // label length (cf. l)
  L = 2*l_label + 1; // modified label length (cf. l')
  //LOG(INFO) << "L: " << L;
  // Generate the new label based on the real label
  // namely, add a blank label between the non-blank label
  // easier to implement further recursion
  real_label_ = new int[l_label];
  for (int i = 0; i < l_label; ++i) {
    real_label_[i] = real_label[i];
  }
  //LOG(INFO) << real_label_[0] << real_label_[1] << real_label_[2] << real_label_[3];
  
  // Dtype** alpha_;
  alpha_ = new Dtype*[T]; // alpha_ has the size of (|t| * |l'|)
  for (int t = 0; t < T; ++t) {
    alpha_[t] = new Dtype[L];
    memset(alpha_[t],0,L*sizeof(Dtype));
  }
   
  alpha_hat_ = new Dtype*[T]; // alpha_hat_ also has the size of (|t| * |l'|)
  for (int t = 0; t < T; ++t) {
    alpha_hat_[t] = new Dtype[L];
    memset(alpha_hat_[t],0,L*sizeof(Dtype));
  }

  // Dtype** beta;
  beta_ = new Dtype*[T];
  for (int t = 0; t < T; ++t) {
    beta_[t] = new Dtype[L];
    memset(beta_[t],0,L*sizeof(Dtype));
  }
  
  beta_hat_ = new Dtype*[T];
  for (int t = 0; t < T; ++t) {
    beta_hat_[t] = new Dtype[L];
    memset(beta_hat_[t],0,L*sizeof(Dtype));
  }
  
  C_ = new Dtype[T];
  memset(C_, Dtype(0), T*sizeof(Dtype));
  D_ = new Dtype[T];
  memset(D_, Dtype(0), T*sizeof(Dtype));
  Q_ = new Dtype[T];
  memset(Q_, Dtype(1), T*sizeof(Dtype));
  
  // Dtype** beta;
  ab_ = new Dtype*[T];
  for (int t = 0; t < T; ++t) {
    ab_[t] = new Dtype[L];
    memset(ab_[t],Dtype(0),L*sizeof(Dtype));
  }

  // forward
  // initialize
  alpha_[0][0] = prob_data[0 * dim + 0];  //y^1_b 
  alpha_[0][1] = prob_data[0 * dim + real_label_[0]]; //y^1_{l_1}
   
  // normalize
  for (int s = 0; s < L; ++s) {
    C_[0] += alpha_[0][s];
  }
  for (int s = 0; s < L; ++s) {
    alpha_hat_[0][s] = alpha_[0][s]/C_[0];
  }
  
  // recurse
  for (int t = 1; t < T; ++t) {
    int start = std::max(0, L-2*(T-t));
    int end = std::min(2*t+2,L);
    for (int s = start; s < L; ++s) {
      int l = (s-1) / 2;
      if (s%2==0) {
        // take care of the boundary values
        if (s==0)
          alpha_[t][s] = alpha_hat_[t-1][s] * prob_data[t*dim+0];
        else
          alpha_[t][s] = ( alpha_hat_[t-1][s] + alpha_hat_[t-1][s-1] ) * prob_data[t*dim+0];
      }
      else if ( (s==1) || (real_label_[l]==real_label_[l-1]) )
        alpha_[t][s] = ( alpha_hat_[t-1][s] + alpha_hat_[t-1][s-1] ) * prob_data[t*dim+real_label_[l]];
      else 
        alpha_[t][s] = ( alpha_hat_[t-1][s] + alpha_hat_[t-1][s-1] + alpha_hat_[t-1][s-2] ) * prob_data[t*dim+real_label_[l]];
    }
    for (int s = start; s < end; ++s) {
      C_[t] += alpha_[t][s];
    }
    // normalize
    for (int s = start; s < end; ++s) {
      alpha_hat_[t][s] = alpha_[t][s]/C_[t];
    }
  }

  // backward
  // initialize
  // should also be equivalent to the following:
  beta_[T-1][L-1] = prob_data[ (T-1)*dim + 0 ];
  beta_[T-1][L-2] = prob_data[ (T-1)*dim + real_label_[l_label-1] ];
  // normalize
  for (int s = L-1; s >= 0; --s) {
    D_[T-1] += beta_[T-1][s];
  }
  for (int s = L-1; s >= 0; --s) {
    beta_hat_[T-1][s] = beta_[T-1][s]/D_[T-1];
  }
  
  // recurse
  for (int t = T-2; t >= 0; --t) {
    int start = std::max(0, L-2*(T-t));
    int end = std::min(2*t+2, L);
    for (int s = end-1; s >= 0; --s) {
      int l = (s - 1) / 2;
      if (s%2==0) {
        // also, take care of the boundary values
        if (s==L-1)
          beta_[t][s] = beta_hat_[t+1][s] * prob_data[t*dim+0];
        else 
          beta_[t][s] = ( beta_hat_[t+1][s] + beta_hat_[t+1][s+1] ) * prob_data[t*dim+0];
      }
      else if ( (s==L-2) or (real_label_[l]==real_label_[l+1]) ) 
        beta_[t][s] = ( beta_hat_[t+1][s] + beta_hat_[t+1][s+1] ) * prob_data[t*dim+real_label_[l]];
      else
        beta_[t][s] = ( beta_hat_[t+1][s] + beta_hat_[t+1][s+1] + beta_hat_[t+1][s+2] ) * prob_data[t*dim+real_label_[l]];
    }
    for (int s = end-1; s >= start; --s) {
      D_[t] += beta_[t][s];
    }
    // normalize 
    for (int s = end-1;  s >= start; --s) {
      beta_hat_[t][s] = beta_[t][s] / D_[t];
    }
  }

  Dtype llForward = 0;
  for (int t = 0; t < T; ++t) {
    llForward -= log(C_[t]);
  }
  top[0]->mutable_cpu_data()[0] = llForward;
 
  Dtype llBackward = 0;
  for (int t = 0; t < T; ++t) {
    llBackward -= log(D_[t]);
  }
  
  for (int t =0; t < T; ++t) {
    for (int ind = 0; ind < t; ++ind)
      Q_[t] *= C_[t];
    for (int ind = t-1; ind < T; ++ind)
      Q_[t] *= D_[t];
  }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    for (int t = 0; t < T; ++t) { 
      for (int s = 0; s < L; ++s) {
        ab_[t][s] = alpha_hat_[t][s] * beta_hat_[t][s];
      }
    }
    
    for (int s = 0; s < L; ++s) {
      if (s%2==0) { 
        for (int t = 0; t < T; ++t) {
          bottom_diff[t*dim + 0] += ab_[t][s];
          ab_[t][s] = ab_[t][s] / prob_data[t*dim + 0];
        } 
      } else {
        int l = (s-1) / 2;
        for (int t = 0; t < T; ++t) {
          bottom_diff[t*dim + real_label_[l]] += ab_[t][s];
          ab_[t][s] = ab_[t][s] / prob_data[t*dim+real_label_[l]];
        }
      } 
    }
    Dtype* absum = new Dtype[T];
    memset(absum,Dtype(0),T*sizeof(Dtype));
    for (int t = 0; t < T; ++t) {
      for (int s = 0; s < L; ++s) {
        absum[t] += ab_[t][s];
      }
    }
   
    for (int t = 0; t < T; ++t){
      if (absum[t]==0)
        LOG(INFO) << "Zero found at time: " << t;
    }
    Dtype llDiff = std::abs(llForward - llBackward);
    if (llDiff>1e-5) 
      LOG(INFO) << "Diff in forward/backward LL: " << llDiff;

    for (int t = 0; t < T; ++t) 
      for (int k = 0; k < dim; ++k) {
        Dtype tmp = prob_data[t*dim+k] * absum[t];
        bottom_diff[t*dim+k] = prob_data[t*dim+k] - bottom_diff[t*dim+k] / tmp;
      }
    delete []absum;
    /*
    for (int t = 0; t < T; ++t) 
      for (int k = 0; k < dim; ++k) {
        pair<std::multimap<int,int>::iterator,std::multimap<int,int>::iterator> ret;
        ret = lab_.equal_range(k);
        Dtype sum = 0;
        for (std::multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it) {
          sum += alpha_hat_[t][it->second]*beta_hat_[t][it->second];
        }
        bottom_diff[t * dim + k] -= Q_[t]/(prob_data[t * dim + k]) * sum;
        //bottom_diff[t * dim + k] = prob_data[t*dim+k] - Q_[t] * sum;
      }
*/  
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_scal(prob_.count(), loss_weight, bottom_diff);
  
  }

  for (int i = 0; i < T; ++i) {
    delete[] alpha_[i];
  }  
  delete []alpha_;
  for (int i = 0; i < T; ++i) {
    delete[] beta_[i];
  }  
  delete []beta_;
  for (int i = 0; i < T; ++i) {
    delete[] alpha_hat_[i];
  }
  delete[]alpha_hat_;
  for (int i = 0; i < T; ++i) {
    delete[] beta_hat_[i];
  } 
  delete []beta_hat_;
  for (int i = 0; i < T; ++i) {
    delete[] ab_[i];
  }
  delete []ab_; 
  delete []C_;
  delete []D_;
  delete []Q_;
  delete []real_label_;
}

INSTANTIATE_CLASS(ConnectionistTemporalClassificationLossLayer);
REGISTER_LAYER_CLASS(ConnectionistTemporalClassificationLoss);

}  // namespace caffe
