def train_loop(T, f):
    for step in tqdm(range(MAX_STEPS)):
    # T optimization
    unfreeze(T); freeze(f)
    avg_gen_opt_step_time = 0
    avg_num_func_eval_per_opt_step = 0
    for t_iter in range(T_ITERS):
       
        generator_opt_step_time = time.time()
        T_opt.zero_grad()
        X = X_sampler.sample(BATCH_SIZE)
        
        t_eval, trajectories = T(X, return_trajectory=True)
        T_X = trajectories[-1]
        
        if regularized:
            # sampling random time for everi image in batch
            rand_ind = torch.randint(len(t_eval), (BATCH_SIZE,))
            random_t = t_eval[rand_ind]
            random_X_batch = trajectories[rand_ind, range(BATCH_SIZE), :, :]

            # T(X, t) -> t - ожидаемые моменты времени (можно ли разные в батче?) odeint -> (либо интерполировать, либо останавливаться в дополнительных точках)
            # T_X = T(X)
            # array все точки во всех временах t?
            # подставить нужный x(t) в регуояризатор
            # альтернативно, заранее задать 10 точек траектории + финальную, для каждого элемента бача сэмплить одну из этих точек для регуляризации 
            # regularization 
            # t = torch.rand(size=(1,)).to(X)
            #TODO sample different time for elem in batch
            # regularizing
            random_t = random_t.detach()
            random_X_batch = random_X_batch.detach()
            value = T.odefunc(random_t, random_X_batch)
            _ , second_derivative = functorch.jvp(func=T.odefunc,
                                                  primals=(random_t, random_X_batch),
                                                  tangents = (torch.tensor([1.]).to(X), value))
            reg_loss = second_derivative.square().mean() 
        else:
            reg_loss = 0
            
        # total_loss
        if COST == 'mse':
            T_loss = F.mse_loss(X, T_X).mean() - f(T_X).mean() + reg_loss
        else:
            raise Exception('Unknown COST')
            
        with torch.no_grad():
            num_func_eval = T.nfe
        T_loss.backward(); T_opt.step()
        with torch.no_grad():
            num_func_eval_per_opt_step = T.nfe - num_func_eval
        
        generator_opt_step_time = time.time() - generator_opt_step_time
        avg_gen_opt_step_time += generator_opt_step_time
        avg_num_func_eval_per_opt_step += num_func_eval_per_opt_step
        
    avg_gen_opt_step_time /= T_ITERS 
    avg_num_func_eval_per_opt_step /= T_ITERS
    # del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()

    # f optimization
    freeze(T); unfreeze(f)
    discriminator_opt_step_time = time.time()
    X = X_sampler.sample(BATCH_SIZE)
    with torch.no_grad():
        T_X = T(X)
    Y = Y_sampler.sample(BATCH_SIZE)
    f_opt.zero_grad()
    f_loss = f(T_X).mean() - f(Y).mean()
    f_loss.backward(); f_opt.step();
    discriminator_opt_step_time = time.time() - discriminator_opt_step_time
    wandb.log({'f_loss' : f_loss.item(),
               'T_loss' : T_loss.item(),
               'reg_loss' : reg_loss.item(),
              'avg_num_func_eval_per_opt_step' : avg_num_func_eval_per_opt_step,
              'avg_gen_opt_step_time' : avg_gen_opt_step_time,
              'discriminator_opt_step_time' : discriminator_opt_step_time
              }, step=step) 
            
    if step % PLOT_INTERVAL == 0:
        print('Plotting')
        clear_output(wait=True)
        
        fig, axes = plot_images(X_fixed, Y_fixed, T)
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_random_images(X_sampler,  Y_sampler, T)
        wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
        wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
        wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        plt.show(fig); plt.close(fig) 
    
    if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
        freeze(T); 
        
        print('Computing FID')
        mu, sigma = get_pushed_loader_stats(T, X_test_sampler.loader)
        fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
        wandb.log({f'FID (Test)' : fid}, step=step)
        # del mu, sigma
        
        torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))
#         torch.save(f.state_dict(), os.path.join(OUTPUT_PATH, f'f_{SEED}_{step}.pt'))
#         torch.save(f_opt.state_dict(), os.path.join(OUTPUT_PATH, f'f_opt_{SEED}_{step}.pt'))
#         torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_{step}.pt')

    