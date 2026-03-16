---
<div class="gn-home">
	<canvas class="gn-web-canvas" aria-hidden="true"></canvas>
	<section class="gn-hero gn-reveal">
		<div class="gn-hero-bg"></div>
		<p class="gn-overline">LET'S CONNECT</p>
		<h1>Get in Touch with Us</h1>
		<p class="gn-subtitle">
			Have a question about Machine Gnostics? Want to partner with us? Ready to submit your data challenge? We'd love to hear from you.
		</p>
	</section>

	<section class="gn-section gn-reveal">
		<h2>Contact Information</h2>
		<p>
			Reach out directly via email or connect with us on social media. Our team monitors inquiries closely and responds within 24-48 hours.
		</p>
	</section>

	<section class="gn-section gn-reveal" id="email-section">
		<h2>Email Us</h2>
		<p>
			For general inquiries, business opportunities, or technical questions:
		</p>
		<div class="gn-email-box">
			<a href="mailto:info.machinegnostics@gmail.com" class="gn-email-link">info.machinegnostics@gmail.com</a>
		</div>
	</section>

	<section class="gn-section gn-reveal" id="contact-form">
		<h2>Send a Message</h2>
		<p>
			Prefer to fill out a form? Use this to tell us more about your inquiry, and we'll get back to you promptly.
		</p>
		<form action="https://formsubmit.co/info.machinegnostics@gmail.com" method="POST" class="gn-contact-form">
			<div class="gn-form-row">
				<div class="gn-form-group">
					<label for="name">Name</label>
					<input type="text" id="name" name="name" required>
				</div>
				<div class="gn-form-group">
					<label for="surname">Surname</label>
					<input type="text" id="surname" name="surname" required>
				</div>
			</div>
			<div class="gn-form-group">
				<label for="email">Email</label>
				<input type="email" id="email" name="email" required>
			</div>
			<div class="gn-form-group">
				<label for="message">Message</label>
				<textarea id="message" name="message" rows="5" required placeholder="Tell us everything..."></textarea>
			</div>
			<!-- Honeypot field to catch bots -->
			<div style="display:none;">
				<input type="text" name="_honey" autocomplete="off">
			</div>
			<!-- FormSubmit settings -->
			<input type="hidden" name="_captcha" value="false">
			<button type="submit" class="md-button md-button--primary">Send Message</button>
		</form>
		<script>
			// Show thank you message after form submission
			document.querySelector('.gn-contact-form').addEventListener('submit', function(e) {
				setTimeout(function() {
					const formElement = document.getElementById('contact-form');
					if (formElement) {
						formElement.innerHTML = '<div style="text-align: center; padding: 2rem;"><h3 style="color: var(--md-primary-fg-color);">✓ Thank You!</h3><p>Your message has been sent successfully. We\'ll get back to you within 24-48 hours.</p><a href="/" class="md-button md-button--primary" style="margin-top: 1rem;">Back to Home</a></div>';
					}
				}, 1000);
			});
		</script>
		<small style="display: block; text-align: center; margin-top: 1rem;">Your message is important to us. We'll respond as soon as possible.</small>
	</section>

	<section class="gn-section gn-reveal">
		<h2>Connect via Discord</h2>
		<p>
			Want to chat with our community in real-time? Join our Discord server where you can discuss Machine Gnostics, ask questions, share projects, and collaborate with other developers.
		</p>
		<div class="gn-actions">
			<a href="https://discord.gg/WMMUaeJe2X" target="_blank" class="md-button md-button--primary">Join Our Discord Community</a>
		</div>
	</section>

	<section class="gn-section gn-reveal">
		<h2>Other Ways to Reach Us</h2>
		<div class="gn-contact-channels">
			<div class="gn-contact-channel">
				<h3>GitHub Issues</h3>
				<p>Report bugs or suggest features directly on our repository.</p>
				<a href="https://github.com/MachineGnostics/machinegnostics/issues" target="_blank">Open an Issue →</a>
			</div>
			<div class="gn-contact-channel">
				<h3>GitHub Discussions</h3>
				<p>Ask questions and discuss with the community in our discussions forum.</p>
				<a href="https://github.com/MachineGnostics/machinegnostics" target="_blank">Join Discussions →</a>
			</div>
			<div class="gn-contact-channel">
				<h3>LinkedIn</h3>
				<p>Connect with us on LinkedIn for company updates and insights.</p>
				<a href="https://www.linkedin.com/company/109036022/" target="_blank">Follow Us →</a>
			</div>
			<div class="gn-contact-channel">
				<h3>YouTube</h3>
				<p>Watch tutorials, webinars, and deep dives into Mathematical Gnostics concepts.</p>
				<a href="https://www.youtube.com/@MachineGnostics" target="_blank">Subscribe →</a>
			</div>
		</div>
	</section>

	<section class="gn-section gn-social gn-reveal">
		<h2>Follow Us Everywhere</h2>
		<p>Stay updated with the latest in Mathematical Gnostics and join our growing community.</p>
		<div class="gn-social-links">
			<a href="https://github.com/MachineGnostics/machinegnostics" target="_blank" title="GitHub">GitHub</a>
			<a href="https://discord.gg/WMMUaeJe2X" target="_blank" title="Discord">Discord</a>
			<a href="https://www.linkedin.com/company/109036022/" target="_blank" title="LinkedIn">LinkedIn</a>
			<a href="https://pypi.org/project/machinegnostics/" target="_blank" title="PyPI">PyPI</a>
			<a href="https://www.instagram.com/machinegnostics/" target="_blank" title="Instagram">Instagram</a>
			<a href="https://www.youtube.com/@MachineGnostics" target="_blank" title="YouTube">YouTube</a>
		</div>
	</section>
</div>

<style>
.gn-email-box {
	margin: 1.5rem 0;
	padding: 1.5rem;
	border-radius: 0.9rem;
	background: linear-gradient(180deg,
				color-mix(in srgb, var(--md-primary-fg-color), transparent 92%) 0%,
				color-mix(in srgb, var(--md-default-bg-color), #111 8%) 100%);
	border: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 78%);
	text-align: center;
}

.gn-email-link {
	display: inline-block;
	font-size: 1.3rem;
	font-weight: 600;
	color: var(--md-primary-fg-color);
	text-decoration: none;
	padding: 0.5rem 1rem;
	transition: all 0.3s ease;
	border-radius: 0.5rem;
}

.gn-email-link:hover {
	background-color: color-mix(in srgb, var(--md-primary-fg-color), transparent 90%);
	text-decoration: underline;
}

.gn-contact-channels {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
	gap: 1rem;
	margin-top: 1.5rem;
}

.gn-contact-channel {
	padding: 1.25rem;
	border-radius: 0.9rem;
	background: linear-gradient(180deg,
				color-mix(in srgb, var(--md-primary-fg-color), transparent 92%) 0%,
				color-mix(in srgb, var(--md-default-bg-color), #111 8%) 100%);
	border: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 78%);
	transition: transform 0.22s ease, border-color 0.22s ease;
}

.gn-contact-channel:hover {
	transform: translateY(-2px);
	border-color: color-mix(in srgb, var(--md-primary-fg-color), transparent 40%);
}

.gn-contact-channel h3 {
	margin-top: 0;
	margin-bottom: 0.5rem;
	color: var(--md-primary-fg-color);
	font-size: 1rem;
}

.gn-contact-channel p {
	margin: 0 0 0.75rem 0;
	font-size: 0.9rem;
	color: var(--md-default-fg-color);
	line-height: 1.5;
}

.gn-contact-channel a {
	font-weight: 600;
	color: var(--md-primary-fg-color);
	text-decoration: none;
	font-size: 0.9rem;
	border-bottom: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 60%);
	transition: border-color 0.3s ease;
}

.gn-contact-channel a:hover {
	border-bottom-color: var(--md-primary-fg-color);
}
</style>
---
