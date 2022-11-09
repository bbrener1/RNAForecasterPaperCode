using Flux
using DrWatson: struct2dict
using Flux: @functor, chunk
using Flux: mse
using Flux.Data: DataLoader
using Logging: with_logger
using Parameters: @with_kw
using Random

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

function reconstuct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(z)
end

function model_loss(encoder, decoder, λ, x, device)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    logp_x_z = -mse(decoder_z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))

    -logp_x_z + kl_q_p + reg
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    λ = 0.01f0              # regularization paramater
    batch_size = 200        # batch size
    sample_size = 10        # sampling size for output
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = false             # use GPU
    input_dim = 10        # image size
    latent_dim = 4          # latent dimension
    hidden_dim = 8        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function trainVAE(data; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim) |> device
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim) |> device

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"

        for (x) in data
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            println(loss)
            train_steps += 1
        end
    end
    return encoder, decoder
end


function encodeData(encoder, data, latentDim)
    out = Matrix{Float32}(undef, latentDim, size(data,2))
    for i=1:size(data,2)
        enc = encoder(data[:,i])
        z = enc[1] + cpu(randn(Float32, size(enc[2]))) .* exp.(enc[2])
        out[:,i] = z
    end
    return out
end


function decodeData(data, decoder, fullDim)
    out = Matrix{Float32}(undef, fullDim, size(data,2))
    for i=1:size(data,2)
        out[:,i] = decoder(data[:,i])
    end
    return out
end
