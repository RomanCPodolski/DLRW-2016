require 'rake/clean'

SOURCE_FILES = Rake::FileList.new('**/*.py') do |fl|
  fl.exclude(/^scratch\//)
  fl.exclude do |f|
    `git ls-files #{f}`.empty?
  end
end

PLOTS = Rake::FileList.new('**/*.png')
SERIALIZATION = Rake::FileList.new('**/*.pkl')
DATASETS = Rake::FileList.new('data/mnist.pkl.gz, data/cifar-10-python.tar')

CLEAN << [ SERIALIZATION, DATASETS ]
CLOBBER << PLOTS

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

    file 'nn/error_tanh.png' => 'nn/best_model_tanh.pkl' do
      %x(python nn/neural_net.py plot error tanh 1>&2)
    end

    file 'nn/repflds_tanh.png' => 'nn/best_model_tanh.pkl' do
      %x(python nn/neural_net.py plot repfds tanh 1>&2)
    end
  end
  task tanh: ['nn/repflds_tanh.png', 'nn/error_tanh.png']

  namespace :sigmoid do
    file 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py train sigmoid 1>&2)
    end

    file 'nn/error_sigmoid.png' => 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py plot error sigmoid 1>&2)
    end

    file 'nn/repflds.png' => 'nn/best_model_sigmoid.pkl' do
      %x(python nn/neural_net.py plot repfds sigmoid 1>&2)
    end
  end
  task sigmoid: ['nn/repfds_sigmoid.png', 'nn/error_sigmoid.png']

  namespace :relu do
    file 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py train relu 1>&2)
    end

    file 'nn/error_relu.png' => 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py plot error relu 1>&2)
    end

    file 'nn/repflds.png' => 'nn/best_model_relu.pkl' do
      %x(python nn/neural_net.py plot repfds relu 1>&2)
    end
  end
  task relu: ['nn/repfds_sigmoid.png', 'nn/error_sigmoid.png']
end
task nn: ['nn:tanh', 'nn:sigmoid', 'nn:relu']

namespace :tsne do
  file 'tsne/bh_tsne.tar.gz' do
    puts 'load '
    %x(wget some_url > tsne/bh_tsne.tar.gz)
  end

  file 'tsne/bh_tsne' => 'tsne/bh_tsne.tar.gz' do
    puts 'unpack '
    %x(tar -xf tsne/bh_tsne.tar.gz -C tsne)
  end

  file 'tsne/bh_tsne/bh_tsne' => 'tsne/bh_tsne' do
    puts 'compile '
    %x(g++ tsne/bh_tsne/sptree.cpp tsne/bh_tsne/tsne.cpp -o tsne/bh_tsne/bh_tsne -O2)
  end

  file 'tsne/data.pkl' => 'tsne/bh_tsne/bh_tsne' do
    puts 'train '
    %x(python tsne/tsne_mnist.py train 1>&2)
  end

  desc 'shit'
  file 'tsne/tsne_mnist.png' do
    puts 'plot '
    %x(python tsne/tsne_mnist.py plot 1>&2)
  end
end
task :tsne ['tsne/tsne_mnist.png']

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
    file 'latent/autoencoder.pkl' do
      %x(python latent/dA.py train 1>&2)
    end

    file 'latent/autoencoderrec.png' => 'latent/autoencoder.pkl' do
      %x(python latent/dA.py plot reconstructions 1>&2)
    end

    file 'latent/autoencoderfilter.png' => 'latent/autoencoder.pkl' do
      %x(python latent/dA.py plot repflds 1>&2)
    end
  end
  task autoencoder: ['latent/autoencoderrec.png', 'latent/autoencoderfilter.png']
end
task latent: ['latent:pca', 'latent:autoencoder']

namespace :kmeans do
  file 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py train 1>&2)
  end

  file 'kmeans/repflds.png' => 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py plot 1>&2)
  end
end

task kmeans: 'kmeans/repflds.png'


task default: [:logreg, :nn, :latent, :tsne, :kmeans]
