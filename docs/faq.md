---
<div class="gn-home">
	<canvas class="gn-web-canvas" aria-hidden="true"></canvas>
	<section class="gn-hero gn-reveal">
		<div class="gn-hero-bg"></div>
		<p class="gn-overline">COMMON QUESTIONS ANSWERED</p>
		<h1>Frequently Asked Questions</h1>
		<p class="gn-subtitle">
			Quick answers to questions about Machine Gnostics, our philosophy, and how we're different from traditional machine learning.
		</p>
	</section>

	<section class="gn-section gn-reveal" id="faq-general">
		<h2>General Questions</h2>
		
		<div class="gn-faq-item">
			<h3>What is Machine Gnostics?</h3>
			<p>
				Machine Gnostics is a Python ecosystem implementing Mathematical Gnostics—a non-statistical approach to data analysis, machine learning, and neural networks. It's built on principles from Riemannian geometry, relativistic mechanics, thermodynamic entropy theory, and deterministic algebra, offering robust alternatives to traditional statistical methods.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>How is Machine Gnostics different from scikit-learn, TensorFlow, or PyTorch?</h3>
			<p>
				Traditional ML frameworks rely on primary on statistical theories that often fail in real-world scenarios. Machine Gnostics uses Mathematical Gnostics principles rooted in geometry and physics instead. Our models work exceptionally well with small datasets, noisy data, and corrupted inputs—situations where traditional methods struggle. 
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Do I need a PhD in mathematics to use Machine Gnostics?</h3>
			<p>
				No. While the underlying theory is rigorous, our APIs are designed to be Pythonic and intuitive. If you can use scikit-learn or pandas, you can use Machine Gnostics. We provide extensive documentation, examples, and a supportive community. The mathematics is there when you need it, but not required for basic usage.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Is Machine Gnostics open source?</h3>
			<p>
				Yes. The core library is open source and available on <a href="https://github.com/MachineGnostics/machinegnostics">GitHub</a>. We believe in transparent, community-driven development. Contributions from developers worldwide are welcome.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Can I use Machine Gnostics in production?</h3>
			<p>
				Absolutely. Machine Gnostics is designed for production environments. Machine Gnostics works well with Mlflow.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What problem does Machine Gnostics solve best?</h3>
			<p>
				Machine Gnostics excels when:
				<ul>
					<li>You have small or limited datasets (traditional ML fails here)</li>
					<li>Your data is noisy or corrupted (we're noise-robust)</li>
					<li>You need explainability and interpretability (we're deterministic)</li>
					<li>You can't afford black-box models (we trace every decision)</li>
					<li>You want to explore new science and technology!</li>
				</ul>
			</p>
		</div>
	</section>

	<section class="gn-section gn-reveal" id="faq-technical">
		<h2>Technical Questions</h2>

		<div class="gn-faq-item">
			<h3>What does "non-statistical" mean?</h3>
			<p>
				Statistical methods assume data comes from probability distributions. Non-statistical methods use geometric and mathematical principles directly. Imagine analyzing the shape of a cloud without assuming it's normally distributed. We analyze the data's intrinsic geometry instead of forcing it into statistical assumptions. This is more robust and works with fewer samples.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>How does Machine Gnostics handle missing data or outliers?</h3>
			<p>
				Our geometric approach naturally handles both. Outliers don't skew our analysis because we look at structural relationships, not statistical moments. Missing data is handled through our gnostic distribution functions which reconstruct patterns without imputation tricks. Our methods are inherently more robust than traditional approaches.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What about computational performance? Is it slower than deep learning frameworks?</h3>
			<p>
				For many tasks, Machine Gnostics is faster because it doesn't require extensive training epochs. Our algorithms are mathematically efficient and often produce results in minutes rather than hours. For very large datasets (millions+ samples), deep learning may have advantages, but our strength lies in scenarios where traditional deep learning can't even start—where you have 10 samples, not 100,000.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Can I combine Machine Gnostics with other libraries like pandas or scikit-learn?</h3>
			<p>
				Yes. Machine Gnostics is designed for integration. It works with NumPy arrays, pandas DataFrames, and follows scikit-learn conventions where possible. You can use our preprocessing tools before Machine Gnostics analysis, or use our results as input to other frameworks. We fit into your existing data science workflow.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Where can I learn more about the underlying mathematics?</h3>
			<p>
				Our complete documentation includes mathematical derivations, proofs, and theoretical foundations. Start with our <a href="https://docs.machinegnostics.com/latest/mg/concepts/">Concepts section</a> for principles, then dive into academic papers and detailed technical docs. We're committed to transparency—nothing is hidden or treated as proprietary magic.
			</p>
		</div>
	</section>

	<section class="gn-section gn-cta gn-reveal" id="faq-next-steps">
		<h2>Ready to Get Started?</h2>
		<p>
			Still have questions? <a href="/contact/">Contact our team</a> or <a href="https://discord.gg/WMMUaeJe2X">join our Discord community</a> where you can ask developers, researchers, and maintainers directly.
		</p>
	</section>

	<section class="gn-section gn-cta gn-reveal">
		<h2>Have Questions We Didn't Answer?</h2>
		<p>
			Get in touch with our team. We're happy to discuss how Machine Gnostics can solve your specific problem.
		</p>
		<div class="gn-actions">
			<a href="/contact/" class="md-button md-button--primary">Send Your Question</a>
		</div>
	</section>
</div>

<style>
.gn-faq-item {
	margin-bottom: 1.5rem;
	padding-bottom: 1.5rem;
	border-bottom: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 85%);
}

.gn-faq-item:last-child {
	border-bottom: none;
}

.gn-faq-item h3 {
	margin: 0 0 0.5rem 0;
	color: var(--md-primary-fg-color);
	font-size: 1.1rem;
}

.gn-faq-item p {
	margin: 0;
	color: var(--md-default-fg-color);
	line-height: 1.6;
}

.gn-faq-item ul {
	margin-top: 0.5rem;
	margin-bottom: 0;
	padding-left: 1.5rem;
}

.gn-faq-item ul li {
	margin-bottom: 0.5rem;
}

.gn-faq-item a {
	color: var(--md-primary-fg-color);
	font-weight: 500;
	text-decoration: none;
	border-bottom: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 60%);
}

.gn-faq-item a:hover {
	border-bottom-color: var(--md-primary-fg-color);
}
</style>
---
