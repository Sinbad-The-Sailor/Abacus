# print('__________________')
# print('0.5882155030362247')
# print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
# print('__________________')
# print('0.014497111303161407')
# print(student_poisson_mix_pdf(3, 0, 1, 0.5, 0.5, 3))
# print('__________________')
# print('0.4879521848980272')
# print(student_poisson_mix_pdf(3, 3, 1, 0.5, 2, 3))
# print('__________________')
# print('0.08690412342325733')
# print(student_poisson_mix_pdf(3, 6, 4, 0.5, 1.5, 3))
# print('__________________')
# print('0.032529717358629986')
# print(student_poisson_mix_pdf(3, 0, 13, 2.5, 0.5, 3))
# print('__________________')
# print('0.0001305029244456048')
# print(student_poisson_mix_pdf(3, 23, 3, 0.1, 0.1, 3))
# print('__________________')

# print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
# print(student_poisson_mix_pdf(3, 0, 1, 0.5, 0.5, 3))
# x = np.linspace(-10, 10, 100)
# y = np.vectorize(student_poisson_mix_cdf)
# z = y(x, 0, 1, 1, 0.5, 5)
# print(z)
# plt.plot(x, z)
# plt.show()
# plt.plot(x, norm.pdf(x, 0, 1), color='red')
# t = time.time()
# print(student_poisson_mix_quantile(0.05, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.1, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# print('-0.07287438558480491')
# t = time.time()
# print(student_poisson_mix_quantile(0.90, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.95, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.99, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# plt.show()