import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:legato_mobile/api/api_exception.dart';
import 'package:legato_mobile/app_services.dart';
import 'package:legato_mobile/theme/linkedin_theme.dart';
import 'package:legato_mobile/widgets/legato_app_bar.dart';

/// Payment step for an accepted consultation. Paymob gateway will plug in here next.
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

  String _formatScheduled(String? iso) {
    if (iso == null || iso.isEmpty) return '—';
    try {
      final dt = DateTime.parse(iso).toLocal();
      return '${dt.day}/${dt.month}/${dt.year} · ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return iso;
    }
  }

  Future<void> _pay() async {
    setState(() => _paying = true);
    try {
      final checkout = _checkout ?? await context.read<AppServices>().legato.checkoutConsultationPayment(widget.consultationId);
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Paymob coming next'),
          content: Text(
            checkout['message']?.toString() ??
                'Payment gateway integration is the next step. Amount due: ${checkout['amount']} ${checkout['currency'] ?? 'EGP'}.',
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
                        'Your lawyer accepted this consultation. Complete payment to confirm the booking. Paymob secure checkout will be enabled in the next update.',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: LegatoLinkedInTheme.textSecondaryAdaptive(context),
                            ),
                      ),
                      const SizedBox(height: 24),
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
