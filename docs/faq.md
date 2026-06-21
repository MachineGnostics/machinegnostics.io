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
			<h3>What is Machine Gnostics in business terms?</h3>
			<p>
				Machine Gnostics is an assumption-free analytics and AI approach designed for small, high-value, noisy datasets. It helps teams extract actionable insights when decisions are costly and data is limited.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What does the term "gnostics" mean in the context of Machine (Mathematical) Gnostics?</h3>
			<p>
				The term "gnostics" comes from the ancient Greek word <em>gnosis</em>, meaning "knowledge" or the "art of knowing." In Machine (Mathematical) Gnostics, it is used purely in this scientific sense of acquiring knowledge, with no intended religious or mystical meaning.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What does “Small Data, Big Impact” mean?</h3>
			<p>
				It means high-impact decisions can be improved even when you only have limited observations. Instead of waiting for massive data collection, you can act earlier on rare events, edge cases, and expensive operational failures.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What is an expensive data problem?</h3>
			<p>
				An expensive data problem is where each data point is costly to generate, collect, or label, and each failure has high business consequences. Examples include industrial failures, quality escapes, safety events, regulated validation cycles, and specialized R&amp;D data.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Why does small-data industry analytics often hit bottlenecks?</h3>
			<p>
				Most pipelines assume large, clean, frequently refreshed datasets. Small-data industries face sparse events, noisy sensors, and long feedback loops. This creates a bottleneck where conventional approaches either overfit or fail to produce trustworthy signals.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>How does Machine Gnostics support lean infrastructure?</h3>
			<p>
				Machine Gnostics is Python-native and practical for teams that cannot maintain heavy AI infrastructure. You can work with constrained datasets and focused workflows, reducing dependency on large-scale data engineering just to get useful results.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Is this “Statistics vs Machine Gnostics”?</h3>
			<p>
				No exactly! Machine (Mathematical) Gnostics is a complementary lens. It helps surface additional structure and decision signals that may be missed by purely statistical pipelines, especially in scarce-data contexts.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>What does assumption-free approach mean?</h3>
			<p>
				Assumption-free means the approach does not depend on strong distribution assumptions to become useful. Instead of forcing your data into idealized statistical forms, Machine Gnostics analyzes underlying structure directly and works with the data reality you have.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Can Machine Gnostics deliver operational excellence in small-data industries?</h3>
			<p>
				Yes. It is built for cases where rare failures and high-stakes decisions matter most. Teams can improve reliability, detect critical patterns earlier, and prioritize actions with greater confidence from limited data.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>Is Machine Gnostics open source, and is a business license available?</h3>
			<p>
				Yes. The core library is open source and available on <a href="https://github.com/MachineGnostics/machinegnostics">GitHub</a>. If your organization needs embedding or proprietary deployment terms, you can <a href="/contact/">contact us</a> to discuss business licensing options.
			</p>
		</div>

		<div class="gn-faq-item">
			<h3>How can I get started quickly?</h3>
			<p>
				Start with the <a href="https://docs.machinegnostics.com/latest/installation/">installation guide</a>, run a worked example, and evaluate Machine Gnostics on your own constrained dataset.
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
