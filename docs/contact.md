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
			<div class="gn-form-group">
				<label for="inquiry-type">Inquiry Type</label>
				<select id="inquiry-type" name="inquiry_type" required>
					<option value="general">General Inquiry</option>
					<option value="partnership">Partnership</option>
					<option value="co-development">Co-Development</option>
					<option value="licensing">Licensing</option>
					<option value="technical">Technical Question</option>
					<option value="community">Community / Media</option>
				</select>
			</div>
			<div class="gn-form-row">
				<div class="gn-form-group">
					<label for="name">First Name</label>
					<input type="text" id="name" name="name" required>
				</div>
				<div class="gn-form-group">
					<label for="surname">Last Name</label>
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
			<!-- FormSubmit settings - unique subject for each email -->
			<input type="hidden" name="_captcha" value="false">
			<input type="hidden" id="subject-field" name="_subject" value="">
			<button type="submit" class="md-button md-button--primary">Send Message</button>
			<p id="form-status" class="gn-form-status" aria-live="polite"></p>
		</form>
		<script>
			// Form submission handler
			const contactForm = document.querySelector('.gn-contact-form');
			if (contactForm) {
				const intentMap = {
					partnership: 'Partnership',
					'co-development': 'Co-Development',
					licensing: 'Licensing',
					technical: 'Technical Question',
					community: 'Community / Media',
					general: 'General Inquiry'
				};

				const params = new URLSearchParams(window.location.search);
				const requestedIntent = params.get('intent');
				const inquiryTypeField = document.getElementById('inquiry-type');
				if (requestedIntent && intentMap[requestedIntent] && inquiryTypeField) {
					inquiryTypeField.value = requestedIntent;
				}

				contactForm.addEventListener('submit', function(e) {
				e.preventDefault();
				
				const form = this;
				const button = form.querySelector('button[type="submit"]');
				const status = document.getElementById('form-status');
				const originalText = button.textContent;
				
				// Validate honeypot
				const honeypot = form.querySelector('input[name="_honey"]');
				if (honeypot && honeypot.value !== '') {
					showErrorMessage('Spam protection was triggered. Please refresh the page and try again.');
					return false;
				}
				
				// Disable button
				button.disabled = true;
				button.textContent = 'Sending...';
				if (status) {
					status.textContent = 'Submitting your message...';
					status.dataset.state = 'pending';
				}
				
				// Prepare form data
				const formData = new FormData(form);
				
				// Create unique subject with sender's name and timestamp
				const name = document.getElementById('name').value;
				const surname = document.getElementById('surname').value;
				const inquiryType = inquiryTypeField ? inquiryTypeField.value : 'general';
				const inquiryLabel = intentMap[inquiryType] || intentMap.general;
				const timestamp = new Date().toLocaleString();
				const subject = `[${inquiryLabel}] ${name} ${surname} - ${timestamp}`;
				
				formData.set('_subject', subject);
				
				// Create abort controller for timeout
				const controller = new AbortController();
				const timeoutId = setTimeout(() => controller.abort(), 8000);
				
				// Submit form
				fetch('https://formsubmit.co/info.machinegnostics@gmail.com', {
					method: 'POST',
					body: formData,
					signal: controller.signal
				})
				.then(response => {
					clearTimeout(timeoutId);
					if (!response.ok) {
						throw new Error(`Submission failed with status ${response.status}`);
					}
					showSuccessMessage();
				})
				.catch(() => {
					clearTimeout(timeoutId);
					button.disabled = false;
					button.textContent = originalText;
					showErrorMessage('We could not confirm delivery. Please try again or email us directly at info.machinegnostics@gmail.com.');
				});
				
				function showSuccessMessage() {
					const formElement = document.getElementById('contact-form');
					if (formElement) {
						formElement.innerHTML = '<div style="text-align: center; padding: 2rem;"><h3 style="color: var(--md-primary-fg-color);">✓ Thank You!</h3><p>Thank you for connecting with us. Your enquiry has been successfully submitted and is being carefully reviewed by our team. We will be in touch shortly to explore how we can best support your objectives. We value your time and your ambition.</p><a href="/" class="md-button md-button--primary" style="margin-top: 1rem;">Back to Home</a></div>';
					}
				}

				function showErrorMessage(message) {
					if (status) {
						status.textContent = message;
						status.dataset.state = 'error';
					}
				}
			});
			}
		</script>
		<small style="display: block; text-align: center; margin-top: 1rem;">Your message is important to us. We'll respond as soon as possible.</small>
	</section>

	<section class="gn-section gn-reveal">
		<h2>Connect via Discord</h2>
		<p>
			Want to chat with our community in real-time? Join our Discord server where you can discuss Machine Gnostics, ask questions, share projects, and collaborate with other developers.
		</p>
		<div class="gn-actions">
			<a href="https://discord.gg/WMMUaeJe2X" target="_blank" rel="noopener noreferrer" class="md-button md-button--primary">Join Our Discord Community</a>
		</div>
	</section>

	<section class="gn-section gn-social gn-reveal">
		<h2>Follow Us Everywhere</h2>
		<p>Stay updated with the latest in Mathematical Gnostics and join our growing community.</p>
		<div class="gn-social-links">
			<a href="https://github.com/MachineGnostics/machinegnostics" target="_blank" rel="noopener noreferrer" title="GitHub">GitHub</a>
			<a href="https://discord.gg/WMMUaeJe2X" target="_blank" rel="noopener noreferrer" title="Discord">Discord</a>
			<a href="https://www.linkedin.com/company/109036022/" target="_blank" rel="noopener noreferrer" title="LinkedIn">LinkedIn</a>
			<a href="https://pypi.org/project/machinegnostics/" target="_blank" rel="noopener noreferrer" title="PyPI">PyPI</a>
			<a href="https://www.instagram.com/machinegnostics/" target="_blank" rel="noopener noreferrer" title="Instagram">Instagram</a>
			<a href="https://www.youtube.com/@MachineGnostics" target="_blank" rel="noopener noreferrer" title="YouTube">YouTube</a>
		</div>
	</section>
</div>

<style>
.gn-form-status {
	min-height: 1.25rem;
	margin: 0.75rem 0 0;
	font-size: 0.85rem;
	text-align: center;
	color: color-mix(in srgb, var(--md-default-fg-color), white 10%);
}

.gn-form-status[data-state="pending"] {
	color: var(--md-default-fg-color);
}

.gn-form-status[data-state="error"] {
	color: #ff8a80;
}

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
	font-size: 0.9rem;
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

.gn-contact-form select {
	width: 100%;
	padding: 0.75rem 0.9rem;
	border-radius: 0.65rem;
	border: 1px solid color-mix(in srgb, var(--md-primary-fg-color), transparent 70%);
	background: color-mix(in srgb, var(--md-default-bg-color), #111 4%);
	color: var(--md-default-fg-color);
	font: inherit;
}

.gn-contact-form select:focus {
	outline: none;
	border-color: color-mix(in srgb, var(--md-primary-fg-color), transparent 25%);
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--md-primary-fg-color), transparent 85%);
}

.gn-contact-channels {
	display: grid;
	grid-template-columns: repeat(2, 1fr);
	gap: 1.5rem;
	margin-top: 1.5rem;
}

@media (max-width: 768px) {
	.gn-contact-channels {
		grid-template-columns: 1fr;
	}
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
