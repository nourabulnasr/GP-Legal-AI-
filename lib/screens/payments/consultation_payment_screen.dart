import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/utils/egypt_time.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

/// Payment step for an accepted consultation via Paymob (test or live).
class ConsultationPaymentScreen extends StatefulWidget {
  const ConsultationPaymentScreen({super.key, required this.consultationId});

  final int consultationId;

  @override
  State<ConsultationPaymentScreen> createState() => _ConsultationPaymentScreenState();
}

class _ConsultationPaymentScreenState extends State<ConsultationPaymentScreen> {
  bool _loading = true;
  bool _paying = false;
  String? _err;
  Map<String, dynamic>? _consultation;
  Map<String, dynamic>? _checkout;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _err = null;
    });
    try {
      final legato = context.read<AppServices>().legato;
      final data = await legato.getConsultation(widget.consultationId);
      final checkout = await legato.checkoutConsultationPayment(widget.consultationId);
      if (!mounted) return;
      setState(() {
        _consultation = data;
        _checkout = checkout;
        _loading = false;
      });
    } on ApiException catch (e) {
      if (mounted) setState(() {
        _err = e.message;
        _loading = false;
      });
    } catch (e) {
      if (mounted) setState(() {
        _err = e.toString();
        _loading = false;
      });
    }
  }

  String _formatScheduled(String? iso) => formatConsultationTimeEgypt(iso);

  Future<void> _pay() async {
    setState(() => _paying = true);
    try {
      final checkout = _checkout ?? await context.read<AppServices>().legato.checkoutConsultationPayment(widget.consultationId);
      if (!mounted) return;

      final ready = checkout['checkout_ready'] == true;
      final paymentUrl = checkout['payment_url']?.toString();

      if (ready && paymentUrl != null && paymentUrl.isNotEmpty) {
        final uri = Uri.parse(paymentUrl);
        final launched = await launchUrl(uri, webOnlyWindowName: '_self', mode: LaunchMode.platformDefault);
        if (!launched && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not open Paymob checkout. Try again or use another browser.')),
          );
        }
        return;
      }

      await showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Paymob not ready'),
          content: Text(
            checkout['message']?.toString() ??
                'Payment gateway is not configured yet. Amount due: ${checkout['amount']} ${checkout['currency'] ?? 'EGP'}.',
          ),
          actions: [
            FilledButton(onPressed: () => Navigator.pop(ctx), child: const Text('OK')),
          ],
        ),
      );
    } on ApiException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    } finally {
      if (mounted) setState(() => _paying = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = _consultation;
    final amount = (_checkout?['amount'] as num?)?.toDouble() ??
        (c?['estimated_total'] as num?)?.toDouble();
    final paid = c?['payment_status']?.toString() == 'paid';
    final active = c?['status']?.toString() == 'active';
    final confirmed = c?['status']?.toString() == 'confirmed';
    return LegatoPageScaffold(
      title: 'Complete payment',
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _err != null
              ? Center(child: Text(_err!, style: TextStyle(color: Theme.of(context).colorScheme.error)))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (paid && confirmed)
                        Card(
                          color: Colors.blue.shade50,
                          child: Padding(
                            padding: const EdgeInsets.all(12),
                            child: Text(
                              'Payment confirmed. Your session starts at ${_formatScheduled(c?['scheduled_at']?.toString())}.',
                              style: TextStyle(color: Colors.blue.shade900, fontWeight: FontWeight.w600),
                            ),
                          ),
                        ),
                      if (active)
                        Card(
                          color: Colors.green.shade50,
                          child: Padding(
                            padding: const EdgeInsets.all(12),
                            child: Text(
                              'Your consultation session is active.',
                              style: TextStyle(color: Colors.green.shade900, fontWeight: FontWeight.w600),
                            ),
                          ),
                        ),
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                c?['lawyer_name']?.toString() ?? 'Lawyer',
                                style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                              ),
                              const SizedBox(height: 8),
                              Text('Duration: ${c?['duration_minutes'] ?? '—'} minutes'),
                              Text('When: ${_formatScheduled(c?['scheduled_at']?.toString())}'),
                              if (c?['hourly_rate'] != null)
                                Text('Rate: ${c!['hourly_rate']}/hr'),
                              const SizedBox(height: 12),
                              Text(
                                amount != null ? '${amount.toStringAsFixed(0)} EGP' : '—',
                                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                      color: LegatoLinkedInTheme.navActiveGold,
                                      fontWeight: FontWeight.w700,
                                    ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        paid
                            ? (confirmed
                                ? 'Return to Alerts. Messaging opens at the scheduled time.'
                                : active
                                    ? 'Your session is active — open Messages to chat.'
                                    : 'Payment received. Messaging opens at the scheduled time.')
                            : 'Your lawyer accepted this consultation. Pay via Paymob secure checkout to confirm the booking.',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                            ),
                      ),
                      const SizedBox(height: 24),
                      if (!paid)
                        FilledButton(
                          onPressed: _paying ? null : _pay,
                          child: _paying
                              ? const SizedBox(
                                  height: 22,
                                  width: 22,
                                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                )
                              : Text(amount != null ? 'Pay ${amount.toStringAsFixed(0)} EGP' : 'Pay now'),
                        ),
                    ],
                  ),
                ),
    );
  }
}
