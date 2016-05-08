namespace :logreg do
  file 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py train 1>&2)
  end

  file 'logreg/repflds.png' => 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py plot repflds 1>&2)
  end

  file 'logreg/error.png' => 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py plot error 1>&2)
  end
end

task logreg: ['logreg/repflds.png', 'logreg/error.png']

namespace :nn do
  namespace :tanh do
    file 'nn/best_model_tanh.pkl' do
      %x(python nn/neural_net.py train tanh 1>&2)
    end

    file 'logreg/error_tanh.png' => 'nn/best_model_tanh.pkl' do
      %x(python nn/neural_net.py plot error tanh 1>&2)
    end

    file 'logreg/repflds_tanh.png' => 'nn/best_model_tanh.pkl' do
      %x(python nn/neural_net.py plot repfds tanh 1>&2)
    end
  end
  task tanh: ['repfds_tanh.png', 'error_tanh.png']

  namespace :sigmoid do
    file 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py train sigmoid 1>&2)
    end

    file 'logreg/error_sigmoid.png' => 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py plot error sigmoid 1>&2)
    end

    file 'logreg/repflds.png' => 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py plot repfds sigmoid 1>&2)
    end
  end
  task sigmoid: ['repfds_sigmoid.png', 'error_sigmoid.png']

  namespace :relu do
    file 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py train relu 1>&2)
    end

    file 'logreg/error_relu.png' => 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py plot error relu 1>&2)
    end

    file 'logreg/repflds.png' => 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py plot repfds relu 1>&2)
    end
  end
  task sigmoid: ['repfds_sigmoid.png', 'error_sigmoid.png']
end
task nn: [:tanh, :sigmoid, :relu]

task :tsne do
  Dir.chdir('tsne')
  %w(python tsne.py 1>&2)
  Dir.chdir('..')
end

namespace :kmeans do
  file 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py train 1>&2)
  end

  file 'kmeans/repflds.png' => 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py plot 1>&2)
  end
end

task kmeans: 'kmeans/repflds.png'

namespace :latent do
  namespace :pca do
    namespace :mnist do
      file 'latent/pca_mnist.pkl' do
        %x(python latent/pca.py mnist train 1>&2)
      end

      file 'latent/scatterplotMNIST.png' => 'latent/pca_mnist.pkl' do
        %x(python latent/pca.py mnist plot 1>&2)
      end
    end

    namespace :cifar do
      file 'latent/pca_cifar.pkl' do
        %x(python latent/pca.py cifar train 1>&2)
      end

      file 'latent/scatterplotCIFAR.png' => 'latent/pca_cifar.pkl' do
        %x(python latent/pca.py cifar plot 1>&2)
      end
    end
  end
  task pca: ['latent/scatterplotMNIST.png', 'latent/scatterplotCIFAR.png']

  namespace :autoencoder do
    file 'autoencoder.pkl' do
      %x(python latent/dA.py train 1>&2)
    end

    file 'latent/autoencoderrec.png' => 'autoencoder.pkl' do
      %x(python latent/dA.py plot reconstructions 1>&2)
    end

    file 'latent/autoencoderfilter.png' => 'autoencoder.pkl' do
      %x(python latent/dA.py plot repflds 1>&2)
    end
  end
  task autoencoder: ['latent/autoencoderrec.png', 'latent/autoencoderfilter.png']
end

task default: [:logreg, :nn, :latent, :tsne, :kmeans]
